import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if sys.platform == "win32":
    import datasets  # noqa: F401  # Load Arrow before torch on Windows (issue #771).
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.quant_linear import install_quantization
from model.quant_policy import (MIXED_SCHEMES, MODEL_SCHEMES, POLICY_SCHEMES, RESIDUAL_POLICY, RESIDUAL_SCHEMES,
                                projection_policy, scheme_artifact_name, validate_resume_policy)
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = last_log_time = time.perf_counter()
    window_tokens = torch.zeros((), dtype=torch.int64, device=args.device)
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        window_tokens.add_(labels[..., 1:].ne(-100).sum())
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        previous_update = ((step - 1) // args.accumulation_steps) * args.accumulation_steps
        next_update = min(previous_update + args.accumulation_steps, iters)
        accumulation_size = next_update - max(previous_update, start_step)
        update = step == next_update
        sync_ctx = model.no_sync() if isinstance(model, DistributedDataParallel) and not update else nullcontext()
        # DDP checks its synchronization flag during forward as well as backward.
        with sync_ctx:
            with autocast_ctx:
                res = model(input_ids, labels=labels)
                loss = (res.loss + res.aux_loss) / accumulation_size
            scaler.scale(loss).backward()

        if update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            if dist.is_initialized():
                dist.all_reduce(window_tokens, op=dist.ReduceOp.SUM)
            valid_tokens = window_tokens.item()  # Wait for this interval's device work.
            now = time.perf_counter()
            tokens_per_second = valid_tokens / (now - last_log_time)
            last_log_time = now
            window_tokens.zero_()
            spend_time = now - start_time
            current_loss = loss.item() * accumulation_size
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min, valid_tokens/s: {tokens_per_second:.0f}')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min, "valid_tokens_per_second": tokens_per_second})

        # Resume files contain optimizer-boundary state, not pending gradients.
        save_due = update and step // args.save_interval > previous_update // args.save_interval
        if (save_due or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            export_dtype = torch.float16 if args.quant_sim == 'none' else torch.float32
            torch.save({k: v.to(device='cpu', dtype=export_dtype) for k, v in state_dict.items()}, ckp + '.tmp')
            os.replace(ckp + '.tmp', ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                          quant_sim=args.quant_sim, export_dtype=export_dtype,
                          projection_policy=projection_policy(args.quant_sim) if args.quant_sim in POLICY_SCHEMES else None,
                          seed=args.seed if args.quant_sim in POLICY_SCHEMES else None,
                          residual_policy=RESIDUAL_POLICY if args.quant_sim in RESIDUAL_SCHEMES else None)
            model.train()
            del state_dict

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, help="训练设备，默认自动选择")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument('--quant_sim', choices=('none', *MODEL_SCHEMES), default='none',
                        help='V100 数值模拟格式；选中时使用 FP32 承载并关闭原生 AMP')
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()
    if args.quant_sim in MIXED_SCHEMES and not args.use_moe:
        parser.error(f'--quant_sim {args.quant_sim} requires --use_moe 1')

    # Bind the rank and connect NCCL before loading model and dataset dependencies.
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
        Logger(f"DDP ready: {dist.get_world_size()} ranks")
    elif args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    from model.model_minimind import MiniMindConfig
    from dataset.lm_dataset import PretrainDataset

    simulating = args.quant_sim != 'none'
    if simulating:
        args.save_weight = f'{args.save_weight}_{scheme_artifact_name(args.quant_sim)}'
        torch.set_float32_matmul_precision('highest')

    # ========== 1. 初始化环境和随机种子 ==========
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    if ckp_data is not None:
        validate_resume_policy(ckp_data, args.quant_sim, seed=args.seed)
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" or simulating else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if simulating:
        install_quantization(model.float(), args.quant_sim)
        Logger(f'quant_sim={args.quant_sim} | FP32 carrier | AMP off | save_weight={args.save_weight}')
        if args.quant_sim in POLICY_SCHEMES:
            Logger(f'projection_policy={projection_policy(args.quant_sim)} | seed={args.seed}')
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(not simulating and args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # The only model buffers are deterministic RoPE tables, identical on every rank.
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                        gradient_as_bucket_view=True)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        setup_seed(args.seed + epoch)
        if train_sampler is None:
            indices = torch.randperm(len(train_ds)).tolist()
        else:
            train_sampler.set_epoch(epoch)
            indices = train_sampler
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

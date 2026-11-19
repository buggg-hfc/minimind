"""Exercise the pretraining accumulation windows without CUDA or a rendezvous."""
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from trainer import train_pretrain as trainer


class RecordingDDP(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.module.weight.fill_(1.)
        self.sync = True
        self.forward_sync = []
        self.backward_sync = []
        self.module.weight.register_hook(lambda grad: self.backward_sync.append(self.sync))

    @contextmanager
    def no_sync(self):
        self.sync = False
        try:
            yield
        finally:
            self.sync = True

    def forward(self, x, labels):
        self.forward_sync.append(self.sync)
        return SimpleNamespace(loss=self.module(x).sum(), aux_loss=x.new_zeros(()))


class AccumulationTest(unittest.TestCase):
    def test_sync_tail_checkpoint_and_tokens(self):
        model = RecordingDDP()
        values = [1., 3., 5.]
        masked_labels = torch.tensor([[0, 1, -100, 2]])
        loader = [(torch.tensor([x]), masked_labels) for x in values]
        report = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(device='cpu', accumulation_steps=2, learning_rate=0.1,
                                   epochs=1, grad_clip=100., log_interval=3, save_interval=1,
                                   save_dir=tmp, save_weight='test', quant_sim='bf16')
            optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
            scaler = torch.amp.GradScaler('cpu', enabled=False)
            with patch.multiple(trainer, create=True, model=model, args=args, scaler=scaler,
                                optimizer=optim, autocast_ctx=nullcontext(),
                                lm_config=SimpleNamespace(use_moe=False, hidden_size=1),
                                DistributedDataParallel=RecordingDDP, Logger=Mock()), \
                 patch.object(trainer, 'get_lr', return_value=args.learning_rate), \
                 patch.object(trainer, 'lm_checkpoint') as save, \
                 patch.object(trainer.time, 'perf_counter', side_effect=[0., 2.]):
                trainer.train_epoch(0, loader, len(loader), wandb=report)

        self.assertEqual(model.forward_sync, [False, True, True])
        self.assertEqual(model.backward_sync, model.forward_sync)
        self.assertAlmostEqual(model.module.weight.item(), 1. - 0.1 * ((1.+3.)/2. + 5.))
        self.assertEqual([c.kwargs['step'] for c in save.call_args_list], [2, 3])
        self.assertEqual(report.log.call_args.args[0]['valid_tokens_per_second'], 3.)


if __name__ == '__main__':
    unittest.main()

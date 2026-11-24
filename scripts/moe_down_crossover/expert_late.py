"""Restore a single-expert prefix and finish the original epoch in pure HiF4."""
import fcntl
from functools import partial
import math
from pathlib import Path

from early_halves import (BRANCHES, VALID_SHA256, _branch, _prepare, _read,
                          _replace_once, adapt_worker)
from expert_prefix import expert_policy, expert_scheme, expert_state


def adapt_expert_worker(source, expert):
    scheme = expert_scheme(expert)
    policy = expert_policy(expert)
    target, = policy["module_overrides"]
    filename = Path(expert_state(expert)).name.replace("_768_moe_", "_{hidden}_moe_")
    source = adapt_worker(source, early_scheme=scheme, early_policy=policy, checkpoint_name=filename)
    source = _replace_once(source, '    if args.branch not in ("hif4", "hif4_down_mxfp8"):',
                           '    if args.branch != "hif4":')
    # All ranks initially configure the late HiF4 policy. At the fork, score the
    # trained prefix with its actual single-expert policy, then restore the one
    # attribute read by the native simulated linear before any training forward.
    call = '''        score = evaluate(raw, valid_ds, device, batch_size=2, branch_dir=out,
                         updates_since_fork=relative, valid_sha256=valid_hash, scheme=args.branch)
'''
    replacement = f'''        score_scheme = args.branch
        if relative == 0:
            target = raw.get_submodule({target!r})
            if target._simulation_scheme != HIF4:
                raise ValueError("Expected pure HiF4 at the late-stage boundary")
            target._simulation_scheme = "mxfp8"
            score_scheme = {scheme!r}
        try:
            score = evaluate(raw, valid_ds, device, batch_size=2, branch_dir=out,
                             updates_since_fork=relative, valid_sha256=valid_hash, scheme=score_scheme)
        finally:
            if relative == 0:
                target._simulation_scheme = HIF4
'''
    source = _replace_once(source, call, replacement)
    compile(source, "fork_worker.py", "exec")
    return source


def _validation(run, relative, expected_scheme):
    path = run / "eval" / f"step-{relative:04d}.json"
    score = _read(path)
    expected = dict(updates_since_fork=relative, eval_scheme=expected_scheme,
                    documents=1024, valid_tokens=236442, batch_size=2, valid_sha256=VALID_SHA256)
    if any(score.get(key) != value for key, value in expected.items()):
        raise ValueError(f"验证范围或推理格式不匹配：{path}")
    ce, ppl = score["mean_ce"], score["ppl"]
    if not (math.isfinite(ce) and ce >= 0 and math.isfinite(ppl) and ppl >= 1
            and math.isclose(math.log(ppl), ce, rel_tol=1e-9, abs_tol=1e-12)):
        raise ValueError(f"验证 CE/PPL 无效：{path}")
    return score


def finish_expert(original, anchor, old_pair, out, expert):
    """Run one H late arm with matched own-format midpoint and pure-H endpoint."""
    original, anchor, old, out = (Path(p).resolve() for p in (original, anchor, old_pair, out))
    if any(out == p or out in p.parents or p in out.parents for p in (original, anchor, old)):
        raise ValueError("后半程输出必须与前缀及历史实验目录隔离")
    scheme = expert_scheme(expert)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / f".{out.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"这组实验已有任务运行：{out}") from exc
        _prepare(original, anchor, old, out, old / "valid.jsonl", early_scheme=scheme,
                 checkpoint=expert_state(expert), worker_adapter=partial(adapt_expert_worker, expert=expert),
                 late_b=None)
        branch = BRANCHES[0]
        _branch(anchor, out, branch, early_description=f"前半程仅第 6 层专家 {expert} 的 down_proj 使用 MXFP8")
        run = out / branch
        midpoint = _validation(run, 0, scheme)
        final = _validation(run, 256, branch)
        last = _read(run / "train" / "update-0256.json")
        if (last.get("relative_update"), last.get("microbatch"), last.get("optimizer_update"),
                last.get("branch")) != (256, 2048, 512, branch):
            raise ValueError(f"训练结束游标不匹配：{run}")
        return dict(midpoint_prefix_own_ce=midpoint["mean_ce"], midpoint_prefix_own_ppl=midpoint["ppl"],
                    mean_ce=final["mean_ce"], ppl=final["ppl"], valid_tokens=final["valid_tokens"],
                    optimizer_update=last["optimizer_update"])

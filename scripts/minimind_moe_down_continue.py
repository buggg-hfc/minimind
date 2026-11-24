"""Continue the paired HiF4 and down-only MXFP8 runs from update 100 to 256.

MINIMIND_FORK_ROOT selects the experiment directory containing hif4-prefix-1024
and pair-1024-to-1424. Both branch checkpoints and their per-rank RNG states
must be available. The full validation report is written to pair-1024-to-2048.
"""
import csv
import errno
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

BASE = Path(os.environ.get("MINIMIND_FORK_ROOT", "/data/lz/minimind-moe-fork-s42")).expanduser().resolve()
OLD = BASE / "pair-1024-to-1424"
NEW = BASE / "pair-1024-to-2048"
PREFIX = BASE / "hif4-prefix-1024"
BRANCHES = ("hif4", "hif4_down_mxfp8")
MODULES = ("fork_worker.py", "fork_eval.py", "fork_metrics.py", "fork_report.py")
VALID_HASH = "72bb2daa3e43457bebd118119f335fcebf6433077ab56ae2ab6f6f9f28f60a04"
START, TARGET, EVERY, WORLD = 100, 256, 20, 8


def patch_worker(text, start=START, target=TARGET):
    """Only permit the handoff target change at the exact committed source cursor."""
    guard = '        if ck["settings"] != settings:\n'
    restart = '    restart = saved.is_file()\n'
    if text.count(guard) != 1 or text.count(restart) != 1:
        raise ValueError("无法识别原训练器的断点校验逻辑")
    strict_resume = restart + (
        '    if not restart:\n'
        '        raise FileNotFoundError(f"续训需要分支自己的 checkpoint.pt：{saved}")\n'
    )
    accept = (
        f'        handoff = (args.updates == {target} and ck.get("relative_update") == {start}\n'
        f'                   and ck.get("step") == base_step + ACCUM * {start}\n'
        f'                   and ck["settings"] == dict(settings, updates={start}))\n'
        '        if ck["settings"] != settings and not handoff:\n'
    )
    patched = text.replace(restart, strict_resume).replace(guard, accept)
    compile(patched, "fork_worker.py", "exec")
    return patched


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, expected):
    result = sha(path)
    if result != expected:
        raise ValueError(f"SHA256 不符：{path}")
    return result


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def link_or_copy(src, dst):
    # Workers commit by replacing their directory entry, never by writing through a link.
    try:
        os.link(src, dst)
    except OSError as e:
        if e.errno not in (errno.EXDEV, errno.EOPNOTSUPP, errno.EPERM, errno.EMLINK):
            raise
        shutil.copy2(src, dst)


def data_files(mid=START):
    for u in range(1, mid + 1):
        yield Path("train") / f"update-{u:04d}.json"
        for rank in range(WORLD):
            yield Path("metrics") / f"rank-{rank:02d}" / f"update-{u:06d}.npz"
    for u in range(0, mid + 1, EVERY):
        yield Path("eval") / f"step-{u:04d}.json"


def verify_summary(path, final):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if [int(row["relative_update"]) for row in rows] != list(range(final + 1)):
        raise ValueError(f"缺少从 0 到 {final} 的完整配对记录：{path}")
    for u, row in enumerate(rows):
        if u % EVERY == 0 or u == final:
            if int(row.get("eval_valid_tokens") or 0) != 236442:
                raise ValueError(f"第 {u} 步缺少完整验证")
            for field in ("eval_ce_a", "eval_ce_b", "eval_ppl_a", "eval_ppl_b"):
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"第 {u} 步验证无效")
    return rows


def source_spec(torch):
    source = read_json(OLD / "experiment.json")
    expected = {"branches": list(BRANCHES), "optimizer_updates": START, "eval_every": EVERY,
                "seed": 42, "world_size": WORLD, "local_batch": 2, "accumulation_steps": 4}
    env = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
           "transformers": version("transformers"), "datasets": version("datasets")}
    if (source["settings"] != expected or source["environment"] != env or
            source["prefix"]["path"] != str(PREFIX) or source["valid_sha256"] != VALID_HASH):
        raise ValueError("源实验的环境、公共前缀或训练日程不匹配")
    verify(OLD / "valid.jsonl", VALID_HASH)
    verify(PREFIX / "inputs.json", source["prefix"]["input_manifest_sha256"])
    for rel, digest in read_json(PREFIX / "inputs.json")["files"].items():
        verify(PREFIX / rel, digest)
    code = {}
    for name in MODULES:
        path = OLD / "scripts" / name
        verify(path, source["workers"][name])
        code[name] = path.read_bytes()
    code["fork_worker.py"] = patch_worker(code["fork_worker.py"].decode("utf-8")).encode("utf-8")
    verify_summary(OLD / "summary.csv", START)
    anchors = {}
    for branch in BRANCHES:
        directory = OLD / branch
        path = directory / "checkpoint.pt"
        c = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        settings = {"branch": branch, "prefix": str(PREFIX), "inputs_sha256": source["prefix"]["input_manifest_sha256"],
                    "valid_sha256": VALID_HASH, "world_size": WORLD, "base_microbatch": 1024,
                    "horizon": 2048, "updates": START, "eval_every": EVERY, "accumulation_steps": 4, "seed": 42}
        if (c.get("settings") != settings or c.get("relative_update") != START or
                (c.get("epoch"), c.get("step"), c.get("world_size")) != (0, 1424, WORLD) or
                not c.get("model") or not c.get("optimizer", {}).get("state")):
            raise ValueError(f"源分支没有提交完整的第 100 次更新：{path}")
        del c
        rng_files = [f"rng/micro-1424-rank-{i}.pt" for i in range(WORLD)]
        anchors[branch] = {rel: sha(directory / rel) for rel in ("checkpoint.pt", *rng_files)}
    spec = {"source_dir": str(OLD), "source_manifest_sha256": sha(OLD / "experiment.json"),
            "source_summary_sha256": sha(OLD / "summary.csv"), "source_anchors": anchors,
            "worker_hashes": {n: hashlib.sha256(b).hexdigest() for n, b in code.items()},
            "valid_sha256": VALID_HASH, "start_update": START, "target_update": TARGET}
    return spec, code


def prepare(torch):
    spec, code = source_spec(torch)
    if NEW.exists():
        if read_json(NEW / "continuation.json") != spec:
            raise ValueError(f"续训目录中的配置或源文件与本次不一致：{NEW}")
        verify(NEW / "valid.jsonl", VALID_HASH)
        for name, expected in spec["worker_hashes"].items():
            verify(NEW / "scripts" / name, expected)
        return
    # Stage both branches completely before any GPU starts.
    with tempfile.TemporaryDirectory(dir=BASE, prefix=".continue-") as tmp:
        stage = Path(tmp) / "run"
        (stage / "scripts").mkdir(parents=True)
        for name, blob in code.items():
            (stage / "scripts" / name).write_bytes(blob)
        link_or_copy(OLD / "valid.jsonl", stage / "valid.jsonl")
        for branch in BRANCHES:
            old = OLD / branch
            dst = stage / branch
            for rel in (*data_files(), *(Path(k) for k in spec["source_anchors"][branch])):
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(old / rel, dst / rel)
        (stage / "continuation.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        stage.rename(NEW)


def run_branch(branch):
    out = NEW / branch
    cmd = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
           f"--nproc-per-node={WORLD}", str(NEW / "scripts/fork_worker.py"), "--prefix", str(PREFIX),
           "--branch-dir", str(out), "--branch", branch, "--valid", str(NEW / "valid.jsonl"),
           "--updates", str(TARGET), "--eval-every", str(EVERY)]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.setdefault("OMP_NUM_THREADS", "1")
    print(f"续训 {branch}，日志 {out / 'console.log'}", flush=True)
    with (out / "console.log").open("a", encoding="utf-8") as log:
        with subprocess.Popen(cmd, cwd=NEW, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace") as p:
            for line in p.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if p.wait():
                raise SystemExit(f"分支中断，可用同一命令续训：{out / 'console.log'}")


def main():
    if sys.argv[1:] in (["--help"], ["-h"]):
        print(__doc__ + f"\n输入 {OLD}\n输出 {NEW}/summary.csv")
        return
    if sys.argv[1:]:
        raise SystemExit("使用 --help 查看运行说明；实验使用固定配置")
    import torch
    if torch.cuda.device_count() != WORLD:
        raise SystemExit("请使用训练环境，并让本机八张 V100 均可见")
    with (BASE / ".continue-1424-to-2048.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("这组续训已在运行")
        prepare(torch)
        for branch in BRANCHES:
            run_branch(branch)
        sys.path.insert(0, str(NEW / "scripts"))
        from fork_report import summarize
        out = NEW / "summary.csv"
        summarize(NEW / BRANCHES[0], NEW / BRANCHES[1], out, strict=True)
        rows = verify_summary(out, TARGET)
        history = verify_summary(OLD / "summary.csv", START)
        if rows[:START + 1] != history:
            raise ValueError("前 100 步的汇总与原实验不一致")
        print(f"PAIR_READY {out}", flush=True)


if __name__ == "__main__":
    main()

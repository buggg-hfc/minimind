"""Endpoint non-inferiority for paired seed differences, with a one-sided 95% t bound."""
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
import tempfile


def _cosine_integral(power, theta):
    """Integral of cos(x)**power from 0 to theta, by reduction for integer powers."""
    sine, cosine = math.sin(theta), math.cos(theta)
    value = sine if power % 2 else theta
    for k in range(2 + power % 2, power + 1, 2):
        value = ((k - 1) * value + sine * cosine ** (k - 1)) / k
    return value


def t95(df):
    """Student's t .95 quantile. t=sqrt(df)*tan(theta) gives density proportional to cos(theta)**(df-1)."""
    if df < 1:
        raise ValueError("Paired inference requires at least two seeds")
    lo, hi = 0.0, math.pi / 2
    # Half the mass is negative; .90 of the positive half gives CDF .95.
    target = 0.9 * _cosine_integral(df - 1, hi)
    for _ in range(54):
        mid = (lo + hi) / 2
        if _cosine_integral(df - 1, mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(df) * math.tan((lo + hi) / 2)


def write_paired_summary(out, deltas, margin_ce):
    deltas = list(deltas)
    if (len(deltas) < 2 or not all(math.isfinite(d) for d in deltas)
            or not math.isfinite(margin_ce) or margin_ce <= 0):
        raise ValueError("Expected at least two finite paired deltas and a positive CE margin")
    avg, sd = mean(deltas), stdev(deltas)
    upper = avg + t95(len(deltas) - 1) * (sd / math.sqrt(len(deltas)))
    result = dict(n_seeds=len(deltas), difference="CE_HiF4_minus_CE_MXFP8", mean_delta_ce=avg,
                  sd_delta_ce=sd, upper_95_one_sided_ce=upper, margin_ce=margin_ce,
                  noninferiority_supported=upper < margin_ce)
    directory = Path(out)
    dest = directory / "paired_summary.json"
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".paired-summary-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result, f, allow_nan=False, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, dest)
    finally:
        Path(tmp).unlink(missing_ok=True)
    verdict = "支持预设非劣界" if upper < margin_ce else "尚未证实满足预设非劣界"
    print(f"HiF4 − MXFP8：平均 ΔCE={avg:+.6f}，单侧 95% t 上界={upper:+.6f}，"
          f"预设界值={margin_ce:g}，{verdict}。", flush=True)
    return dest

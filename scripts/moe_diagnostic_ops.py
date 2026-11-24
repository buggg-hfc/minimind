"""Component error probes on common local operands from a BF16 MiniMind MoE pass."""
import hashlib

import numpy as np
import torch

from model.quant_linear import _bf16_value, gemm
from model.quant_policy import projection_policy


SIGN_CLASSES = ("same_direction_shrink", "same_direction_equal", "same_direction_expand",
                "zeroed", "sign_flip")


class _PairedStats:
    """Pool finite sampled coordinate pairs across events, with centered variances."""

    def __init__(self):
        self.calls = self.input_rows = self.output_elements = self.sampled_count = 0
        self.min_input_rows = self.max_input_rows = None
        self.finite_count = self.nonzero_count = self.zero_to_nonzero = 0
        self.signs = np.zeros(len(SIGN_CLASSES), dtype=np.int64)
        self.error_sum = self.abs_sum = self.relative_sum = self.ulp_sum = self.hist_kl_sum = 0.0
        self.ref_mean = self.ref_m2 = self.est_mean = self.est_m2 = 0.0

    def add(self, reference, estimate, input_rows, output_elements):
        self.calls += 1
        self.input_rows += input_rows
        self.min_input_rows = input_rows if self.min_input_rows is None else min(self.min_input_rows, input_rows)
        self.max_input_rows = input_rows if self.max_input_rows is None else max(self.max_input_rows, input_rows)
        self.output_elements += output_elements
        self.sampled_count += reference.size
        finite = np.isfinite(reference) & np.isfinite(estimate)
        ref = reference[finite].astype(np.float64, copy=False)
        est = estimate[finite].astype(np.float64, copy=False)
        if not ref.size:
            return
        n0, n1 = self.finite_count, ref.size
        self.finite_count += n1
        # Use shared bins for the paired values of this invocation and stage.
        # The saved value is the finite-sample-count-weighted mean of event KLs.
        lo, hi = min(ref.min(), est.min()), max(ref.max(), est.max())
        if lo != hi:
            r_counts = np.histogram(ref, bins=128, range=(lo, hi))[0] + 0.5
            e_counts = np.histogram(est, bins=128, range=(lo, hi))[0] + 0.5
            event_kl = (r_counts * np.log(r_counts / e_counts)).sum() / r_counts.sum()
            self.hist_kl_sum += max(0.0, event_kl) * n1
        for key, values in (("ref", ref), ("est", est)):
            prev_mean = getattr(self, key + "_mean")
            mean = values.mean()
            shift = mean - prev_mean
            setattr(self, key + "_mean", prev_mean + shift * (n1 / self.finite_count))
            m2 = np.square(values - mean).sum() + shift**2 * (n0 * n1 / self.finite_count)
            setattr(self, key + "_m2", getattr(self, key + "_m2") + m2)

        dr = est - ref
        abs_dr = np.abs(dr)
        r, e = np.abs(ref), np.abs(est)
        nz = r != 0
        ulp = np.maximum(np.exp2(np.frexp(r)[1] - 8.0), 2.0**-133)
        ulp[~nz] = 2.0**-133
        self.error_sum += dr.sum()
        self.abs_sum += abs_dr.sum()
        self.relative_sum += (abs_dr[nz] / r[nz]).sum()
        self.ulp_sum += (abs_dr / ulp).sum()
        self.nonzero_count += int(nz.sum())
        self.zero_to_nonzero += int((~nz & (e != 0)).sum())
        same = nz & (e != 0) & (np.signbit(ref) == np.signbit(est))
        masks = (same & (e < r), same & (e == r), same & (e > r),
                 nz & (e == 0), nz & (e != 0) & ~same)
        self.signs += [np.count_nonzero(m) for m in masks]

    def result(self):
        n, nz = self.finite_count, self.nonzero_count
        result = dict(calls=self.calls, input_rows=self.input_rows,
                      min_input_rows=self.min_input_rows, max_input_rows=self.max_input_rows,
                      output_elements=self.output_elements,
                      sampled_count=self.sampled_count, finite_count=n, nonfinite_count=self.sampled_count - n,
                      ref_zero_count=n - nz, ref_nonzero_count=nz, ref_zero_est_nonzero_count=self.zero_to_nonzero,
                      signed_mean=self.error_sum / n if n else None, mae=self.abs_sum / n if n else None,
                      ulp_mean=self.ulp_sum / n if n else None, relative_mae=self.relative_sum / nz if nz else None,
                      operator_hist_kl=self.hist_kl_sum / n if n else None,
                      variance_retention=self.est_m2 / self.ref_m2 if self.ref_m2 else None)
        for label, count in zip(SIGN_CLASSES, self.signs):
            result[label + "_count"] = int(count)
            result[label + "_rate"] = int(count) / nz if nz else None
        return result

    def merge(self, other):
        """Combine disjoint layer/expert samples; each event keeps its own KL bins."""
        n0, n1 = self.finite_count, other.finite_count
        if n1:
            for key in ("ref", "est"):
                mean = getattr(self, key + "_mean")
                shift = getattr(other, key + "_mean") - mean
                setattr(self, key + "_mean", mean + shift * (n1 / (n0 + n1)))
                m2 = getattr(other, key + "_m2") + shift**2 * (n0 * n1 / (n0 + n1))
                setattr(self, key + "_m2", getattr(self, key + "_m2") + m2)
        for field in ("calls", "input_rows", "output_elements", "sampled_count", "finite_count",
                      "nonzero_count", "zero_to_nonzero", "error_sum", "abs_sum", "relative_sum",
                      "ulp_sum", "hist_kl_sum"):
            setattr(self, field, getattr(self, field) + getattr(other, field))
        self.signs += other.signs
        if other.calls:
            self.min_input_rows = (other.min_input_rows if self.min_input_rows is None
                                   else min(self.min_input_rows, other.min_input_rows))
            self.max_input_rows = (other.max_input_rows if self.max_input_rows is None
                                   else max(self.max_input_rows, other.max_input_rows))


class OperatorProbe:
    """Replay FWD, dX and dW using the BF16 reference's local X, W and upstream dY.

    Attach before a BF16 forward and keep the context active through backward. The model
    is never requantized. GEMMs use full operands including padding, expert routing and
    native dY tensor scales. Reduce up to max_elements uniform output coordinates per call.
    Metrics pool these sampled coordinates. Raw is the pre-output-round GEMM, applied is its
    BF16-rounded value. All probed MiniMind projections are bias-free. Rows with no calls
    identify experts that did not receive any routed tokens during the pass.
    """

    def __init__(self, model, scheme, layers, *, max_elements=4096, seed=42):
        if max_elements <= 0:
            raise ValueError("max_elements must be positive")
        if not model.config.use_moe or scheme == "none":
            raise ValueError("OperatorProbe requires a simulated MiniMind MoE scheme")
        self.seed, self.max_elements, self.scheme = seed, max_elements, scheme
        self.modules = {}
        self.stats = {}
        self.handles = []
        policy = projection_policy(scheme)
        for layer in sorted(set(layers)):
            if not 0 <= layer < len(model.model.layers):
                raise ValueError(f"Invalid layer {layer}")
            block = model.model.layers[layer]
            for leaf in ("q_proj", "k_proj", "v_proj", "o_proj"):
                self._add(layer, None, leaf, getattr(block.self_attn, leaf), policy["attention"])
            for expert, ffn in enumerate(block.mlp.experts):
                for leaf in ("up_proj", "gate_proj", "down_proj"):
                    fmt = policy["ffn"][leaf] if isinstance(policy["ffn"], dict) else policy["ffn"]
                    self._add(layer, expert, leaf, getattr(ffn, leaf), fmt)
        if not self.modules:
            raise ValueError("Select at least one operator probe layer")

    def _add(self, layer, expert, leaf, module, fmt):
        if module.bias is not None or getattr(module, "_simulation_scheme", None) != "bf16":
            raise ValueError("Install BF16 simulation on the reference's bias-free projections first")
        prefix = "self_attn" if expert is None else f"mlp.experts.{expert}"
        path = f"model.layers.{layer}.{prefix}.{leaf}"
        metadata = dict(layer=layer, expert=expert, projection=leaf, module=path,
                        scheme=self.scheme, primitive=fmt)
        self.modules[path] = module, metadata
        for op in ("FWD", "dX", "dW"):
            for stage in ("raw", "applied"):
                self.stats[path, op, stage] = _PairedStats()

    def _observe(self, path, op, lhs, rhs, rows):
        population = lhs.shape[0] * rhs.shape[1]
        if population == 0:
            return
        idx = None
        if population > self.max_elements:
            call = self.stats[path, op, "raw"].calls
            key = f"{self.seed}|{path}|{op}|{call}|{lhs.shape[0]}|{rhs.shape[1]}"
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
            idx = torch.as_tensor(np.sort(rng.choice(population, self.max_elements, replace=False)),
                                  device=lhs.device, dtype=torch.long)

        def sample(fmt):
            full = gemm(lhs, rhs, fmt, lhs_is_dy=op != "FWD").reshape(-1)
            selected = full if idx is None else full.index_select(0, idx)
            return selected.cpu().numpy(), _bf16_value(selected).cpu().numpy()

        reference = sample("bf16")
        fmt = self.modules[path][1]["primitive"]
        estimate = reference if fmt == "bf16" else sample(fmt)
        for stage, ref, est in zip(("raw", "applied"), reference, estimate):
            self.stats[path, op, stage].add(ref, est, rows, population)

    def _hook(self, path):
        @torch.no_grad()
        def forward(module, args, output):
            x = args[0].detach().reshape(-1, module.in_features)
            rows = x.shape[0]
            self._observe(path, "FWD", _bf16_value(x), _bf16_value(module.weight).T, rows)
            if output.requires_grad:
                @torch.no_grad()
                def backward(grad):
                    dy = _bf16_value(grad.detach().reshape(-1, module.out_features))
                    self._observe(path, "dX", dy, _bf16_value(module.weight), rows)
                    self._observe(path, "dW", dy.T, _bf16_value(x), rows)
                output.register_hook(backward)
        return forward

    def __enter__(self):
        if self.handles:
            raise RuntimeError("OperatorProbe is already active")
        for path, (module, _) in self.modules.items():
            self.handles.append(module.register_forward_hook(self._hook(path)))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def rows(self):
        return [dict(self.modules[path][1], gemm=op, stage=stage, **stats.result())
                for (path, op, stage), stats in self.stats.items()]

    def component_rows(self):
        """Pool sampled coordinates across layers and experts for each projection."""
        groups = {}
        for (path, op, stage), stats in self.stats.items():
            meta = self.modules[path][1]
            key = meta["projection"], meta["primitive"], op, stage
            if key not in groups:
                groups[key] = _PairedStats()
            groups[key].merge(stats)
        return [dict(scheme=self.scheme, projection=leaf, primitive=fmt, gemm=op, stage=stage,
                     **stats.result())
                for (leaf, fmt, op, stage), stats in groups.items()]

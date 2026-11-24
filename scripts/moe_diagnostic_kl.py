"""Paired next-token / router diagnostics and learned-weight histogram divergences."""
import math
import re

import numpy as np
import torch

from model.quant_linear import install_quantization
from model.quant_policy import MODEL_SCHEMES


def _backbone(model, tokens, keep, *, save_dispatch=False, force_dispatch=None):
    """Store only the first gate invocation (the native top-1 STE calls it twice)."""
    routes, dispatch = {}, {}
    handles = []

    def capture(layer):
        def save(_module, _args, logits):
            forced = None if force_dispatch is None else force_dispatch[layer]
            # A second router call implements the native STE. Both calls must select the
            # same expert. With normalized top-1, only the chosen ID affects evaluation.
            replacement = (None if forced is None else torch.zeros_like(logits).scatter_(1, forced, 1.0))
            if layer in routes:
                return replacement
            # Mirror the native GPU topk on its original shape, including its tie behavior.
            scores = logits.softmax(-1)
            chosen = scores.topk(1, dim=-1, sorted=False).indices
            if save_dispatch:
                dispatch[layer] = chosen
            proposed = chosen.reshape_as(tokens)[:, :-1][keep].cpu()
            actual = proposed if forced is None else forced.reshape_as(tokens)[:, :-1][keep].cpu()
            valid_logits = logits.reshape(*tokens.shape, -1)[:, :-1][keep]
            valid_scores = scores.reshape(*tokens.shape, -1)[:, :-1][keep]
            margins = valid_scores.topk(2, dim=-1, sorted=True).values
            routes[layer] = (actual, proposed, (margins[:, 0] - margins[:, 1]).cpu(),
                             valid_logits.cpu().double().log_softmax(-1))
            return replacement
        return save

    try:
        for layer, block in enumerate(model.model.layers):
            if block.mlp.config.num_experts_per_tok != 1 or block.mlp.config.num_experts < 2:
                raise ValueError("Router diagnostics require native top-1 MoE with at least two experts")
            handles.append(block.mlp.gate.register_forward_hook(capture(layer)))
        hidden, _, _ = model.model(tokens, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return hidden, routes, dispatch


def _distribution_kl(logp, logq):
    return (logp.exp() * (logp - logq)).sum(dim=-1)


def _merge_routes(accum, ref, cand):
    if ref.keys() != cand.keys():
        raise ValueError("Different MoE layers in the two forwards")
    for layer in ref:
        rid, _, rm, rp = ref[layer]
        cid, proposed, cm, cp = cand[layer]
        if rp.shape != cp.shape:
            raise ValueError(f"Different router shapes in layer {layer}")
        experts = rp.shape[-1]
        if layer not in accum:
            accum[layer] = dict(tokens=0, flips=0, proposed_flips=0, ref_margin=0., cand_margin=0.,
                                flipped_margin=0., stable_margin=0., kl=0.,
                                margin_counts=torch.zeros(4, dtype=torch.long),
                                margin_flips=torch.zeros(4, dtype=torch.long),
                                ref_load=torch.zeros(experts, dtype=torch.long),
                                cand_load=torch.zeros(experts, dtype=torch.long))
        s = accum[layer]
        flipped = rid != proposed
        s["tokens"] += rid.numel()
        s["flips"] += int((rid != cid).sum())
        s["proposed_flips"] += int(flipped.sum())
        bins = torch.bucketize(rm, rm.new_tensor((.01, .05, .20)), right=True)
        s["margin_counts"] += torch.bincount(bins, minlength=4)
        s["margin_flips"] += torch.bincount(bins[flipped], minlength=4)
        s["ref_margin"] += rm.double().sum().item()
        s["cand_margin"] += cm.double().sum().item()
        s["flipped_margin"] += rm[flipped].double().sum().item()
        s["stable_margin"] += rm[~flipped].double().sum().item()
        s["kl"] += _distribution_kl(rp, cp).sum().item()
        s["ref_load"] += torch.bincount(rid, minlength=experts)
        s["cand_load"] += torch.bincount(cid, minlength=experts)


def _finish_routes(accum):
    routing, loads, bins = [], [], []
    for layer, s in sorted(accum.items()):
        n, f = s["tokens"], s["proposed_flips"]
        routing.append(dict(layer=layer, valid_tokens=n, top1_flips=s["flips"], top1_flip_rate=s["flips"]/n,
                            proposed_top1_flips=f, proposed_top1_flip_rate=f/n,
                            router_kl=max(0., s["kl"]/n), mean_bf16_margin=s["ref_margin"]/n,
                            mean_candidate_margin=s["cand_margin"]/n,
                            mean_bf16_margin_proposed_flip=s["flipped_margin"]/f if f else None,
                            mean_bf16_margin_proposed_stable=s["stable_margin"]/(n-f) if f < n else None))
        edges = (0., .01, .05, .20, 1.)
        for i, (count, flips) in enumerate(zip(s["margin_counts"].tolist(), s["margin_flips"].tolist())):
            bins.append(dict(layer=layer, margin_lower=edges[i], margin_upper=edges[i+1], tokens=count,
                             proposed_flips=flips, proposed_flip_rate=flips/count if count else None))
        for expert, (a, b) in enumerate(zip(s["ref_load"].tolist(), s["cand_load"].tolist())):
            loads.append(dict(layer=layer, expert=expert, valid_tokens=n, bf16_tokens=a,
                              candidate_tokens=b, bf16_share=a/n, candidate_share=b/n))
    return routing, loads, bins


@torch.no_grad()
def compare_forwards(model, batches, scheme, *, reference_model=None, token_chunk=64, lock_routes=False):
    """One matched token sample. The gate mask selects predictor positions with a valid next label.

    Default: same learned weights for the two passes. An explicit reference model is evaluated
    with BF16 on the same tokens. Retains separate hidden states and chunks full-vocab logits.
    On exit both models have BF16 simulation installed, and their training flags are restored.
    """
    if token_chunk < 1 or scheme not in MODEL_SCHEMES:
        raise ValueError("Expected a positive token_chunk and a simulated model scheme")
    ref = model if reference_model is None else reference_model
    device = next(model.parameters()).device
    if lock_routes and (ref is not model or not model.config.norm_topk_prob or model.config.num_experts_per_tok != 1):
        raise ValueError("Locking routes requires identical weights and normalized top-1 MoE")
    if next(ref.parameters()).device != device:
        raise ValueError("Place the paired models on the same device")
    was_training = [(m, m.training) for m in dict.fromkeys((model, ref))]
    docs, routes = [], {}
    try:
        for m, _ in was_training:
            m.eval()
        if ref is not model:
            install_quantization(ref, "bf16")
            install_quantization(model, scheme)
        for tokens, labels in batches:
            tokens, labels = tokens.to(device), labels.to(device)
            if tokens.ndim != 2 or labels.shape != tokens.shape:
                raise ValueError("Expected matching [batch, sequence] tokens and labels")
            targets = labels[:, 1:]
            keep = targets != -100
            if ref is model:
                install_quantization(ref, "bf16")
            rh, rr, dispatch = _backbone(ref, tokens, keep, save_dispatch=lock_routes)
            if ref is model and scheme == "bf16":
                ch, cr = rh, rr
            else:
                if ref is model:
                    install_quantization(model, scheme)
                ch, cr, _ = _backbone(model, tokens, keep, force_dispatch=dispatch if lock_routes else None)
            _merge_routes(routes, rr, cr)
            for i in range(tokens.shape[0]):
                gold = targets[i][keep[i]]
                a = rh[i, :-1][keep[i]]
                b = ch[i, :-1][keep[i]]
                sums = torch.zeros(3, dtype=torch.float64, device=device)
                for pos in range(0, gold.numel(), token_chunk):
                    sl = slice(pos, pos+token_chunk)
                    # The simulated output head is BF16 in all experiment schemes.
                    lp = ref.lm_head(a[sl]).log_softmax(-1, dtype=torch.float64)
                    lq = model.lm_head(b[sl]).log_softmax(-1, dtype=torch.float64)
                    y = gold[sl, None]
                    sums[0] -= lp.gather(1, y).sum()
                    sums[1] -= lq.gather(1, y).sum()
                    sums[2] += _distribution_kl(lp, lq).sum()
                p, q, k = sums.tolist()
                if not all(math.isfinite(v) for v in (p, q, k)) or k < -1e-10:
                    raise FloatingPointError("Invalid cross entropy or token KL")
                count = gold.numel()
                docs.append(dict(document=len(docs), valid_tokens=count, bf16_nll_sum=p,
                                 candidate_nll_sum=q, token_kl_sum=max(k, 0.),
                                 mean_ce_bf16=p/count if count else None,
                                 mean_ce_candidate=q/count if count else None,
                                 delta_ce=(q-p)/count if count else None,
                                 next_token_kl=max(k, 0.)/count if count else None))
    finally:
        for m, flag in was_training:
            install_quantization(m, "bf16")
            m.train(flag)
    n = sum(d["valid_tokens"] for d in docs)
    if not n:
        raise ValueError("No valid next-token predictions")
    p, q, k = (math.fsum(d[key] for d in docs)/n
               for key in ("bf16_nll_sum", "candidate_nll_sum", "token_kl_sum"))
    routing, loads, bins = _finish_routes(routes)
    return dict(summary=dict(valid_tokens=n, mean_ce_bf16=p, mean_ce_candidate=q, ppl_bf16=math.exp(p),
                             ppl_candidate=math.exp(q), next_token_kl=k, lock_routes=lock_routes),
                documents=docs, routing=routing, expert_loads=loads, margin_bins=bins)


def weight_kl_rows(reference_state, candidate_state, *, layers=None):
    """Compare *learned* weights, one histogram per named tensor (wgbias convention).

    Uses the joint min/max, 128 equal-width bins, and a 0.5 pseudocount in every bin.
    layers=None includes all weights and counts identical tied embed/head once. Passing layer
    indices selects decoder tensors only. States should be CPU-mapped native MiniMind exports.
    """
    layer_set = None if layers is None else set(layers)
    names = [name for name in reference_state if name.endswith(".weight")]
    if not names or set(names) != {name for name in candidate_state if name.endswith(".weight")}:
        raise ValueError("Different or empty sets of learned weights")
    embed, head = "model.embed_tokens.weight", "lm_head.weight"
    if layer_set is None and embed in names and head in names and all(
        torch.equal(st[embed], st[head]) for st in (reference_state, candidate_state)
    ):
        names.remove(head)
    rows = []
    for name in names:
        layer_match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        expert_match = re.search(r"(?:^|\.)experts\.(\d+)\.", name)
        layer = int(layer_match[1]) if layer_match else None
        if layer_set is not None and layer not in layer_set:
            continue
        a, b = reference_state[name], candidate_state[name]
        if a.shape != b.shape or a.numel() == 0:
            raise ValueError(f"Invalid weight shape: {name}")
        x, y = (t.detach().float().cpu().numpy().reshape(-1) for t in (a, b))
        lo = float(np.minimum(x.min(), y.min()))
        hi = float(np.maximum(x.max(), y.max()))
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError(f"Nonfinite weights: {name}")
        fwd = rev = 0.
        if lo != hi:
            def histogram(v):
                h = np.zeros(128, dtype=np.float64)
                for start in range(0, v.size, 1 << 20):
                    z = v[start:start+(1 << 20)].astype(np.float64, copy=False)
                    h += np.histogram(z, bins=128, range=(lo, hi))[0]
                return (h + 0.5) / (v.size + 64)
            ph, qh = histogram(x), histogram(y)
            ratio = np.log(ph) - np.log(qh)
            fwd, rev = max(0., float(ph @ ratio)), max(0., float(qh @ -ratio))
        rows.append(dict(module=name.rsplit(".", 1)[0], layer=layer,
                         expert=int(expert_match[1]) if expert_match else None, component=name.split(".")[-2],
                         elements=a.numel(), kl_forward=fwd, kl_reverse=rev))
    return rows

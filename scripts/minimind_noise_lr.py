"""Perturb the early learning-rate schedule using the native trainer's global step."""


def with_linear_warmup(native_get_lr, warmup_microbatches=104):
    """Ramp to the unshifted native LR at the join; step counts global microbatches."""
    if warmup_microbatches < 1:
        raise ValueError("warmup_microbatches must be positive")

    def get_lr(current_step, total_steps, lr):
        join = min(warmup_microbatches, total_steps)
        if current_step >= join:
            return native_get_lr(current_step, total_steps, lr)
        return native_get_lr(join, total_steps, lr) * (current_step / join)

    return get_lr

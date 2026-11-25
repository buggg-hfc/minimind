"""Apply the same first-update microbatch swap to each native sampler shard."""
from itertools import chain, islice


def swapped_first_pair_sampler(native_sampler):
    """Wrap MiniMind's SkipBatchSampler, preserving its length and resume cursor."""
    class FirstPairSwapped(native_sampler):
        def __iter__(self):
            if self.skip_batches >= 2:
                # The native and perturbed streams have the same suffix here.
                yield from super().__iter__()
                return

            batches = iter(native_sampler(self.sampler, self.batch_size, skip_batches=0))
            first_pair = list(islice(batches, 2))
            ordered = chain(reversed(first_pair), batches)
            yield from islice(ordered, max(0, self.skip_batches), None)

    return FirstPairSwapped

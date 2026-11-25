import unittest

from torch.utils.data import DistributedSampler

from scripts.minimind_noise_swap import swapped_first_pair_sampler
from trainer.trainer_utils import SkipBatchSampler


class NoiseSwapTest(unittest.TestCase):
    def test_shards_swap_the_same_two_global_microbatches_before_skipping(self):
        swapped = swapped_first_pair_sampler(SkipBatchSampler)
        dataset = range(128)
        global_native = []
        global_swapped = []
        for rank in range(8):
            indices = DistributedSampler(dataset, num_replicas=8, rank=rank, seed=0)
            native = list(SkipBatchSampler(indices, batch_size=2))
            expected = [native[1], native[0], *native[2:]]
            observed = list(swapped(indices, 2))
            self.assertEqual(observed, expected)
            self.assertEqual(sorted(sum(native[:4], [])), sorted(sum(observed[:4], [])))
            global_native.append(native)
            global_swapped.append(observed)
            for cursor in range(len(native) + 2):
                with self.subTest(rank=rank, cursor=cursor):
                    restarted = swapped(indices, batch_size=2, skip_batches=cursor)
                    self.assertEqual(list(restarted), expected[cursor:])
                    self.assertEqual(len(restarted), len(expected[cursor:]))
        global_native = list(zip(*global_native))
        self.assertEqual(list(zip(*global_swapped)), [global_native[1], global_native[0], *global_native[2:]])

    def test_partial_batch_and_short_inputs(self):
        swapped = swapped_first_pair_sampler(SkipBatchSampler)
        for n, expected in ((0, []), (1, [[0]]), (2, [[0, 1]]), (3, [[2], [0, 1]]),
                            (5, [[2, 3], [0, 1], [4]])):
            for cursor in range(4):
                with self.subTest(n=n, cursor=cursor):
                    sampler = swapped(range(n), 2, cursor)
                    self.assertEqual(list(sampler), expected[cursor:])
                    self.assertEqual(len(sampler), len(expected[cursor:]))


if __name__ == "__main__":
    unittest.main()

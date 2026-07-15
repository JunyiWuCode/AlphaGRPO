import unittest

from rewards.sglang_spectrareward import mean_target_logprob


class MeanTargetLogprobTest(unittest.TestCase):
    def test_finds_target_and_excludes_end_token(self):
        entries = [
            (-9.0, 7, None),
            (-0.2, 10, None),
            (-0.4, 11, None),
            (-3.0, 99, None),
        ]
        score = mean_target_logprob(entries, [10, 11, 99], {99})
        self.assertAlmostEqual(score, -0.3)

    def test_uses_last_matching_target(self):
        entries = [
            (-5.0, 10, None),
            (-5.0, 11, None),
            (-0.1, 10, None),
            (-0.3, 11, None),
        ]
        score = mean_target_logprob(entries, [10, 11])
        self.assertAlmostEqual(score, -0.2)

    def test_rejects_mismatched_tokenization(self):
        with self.assertRaisesRegex(ValueError, "do not contain"):
            mean_target_logprob([(-1.0, 1, None)], [2])


if __name__ == "__main__":
    unittest.main()

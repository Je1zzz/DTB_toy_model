import tempfile
import unittest
from pathlib import Path

import numpy as np

from vbt.evaluation.ev import epileptogenicity_value, source_onsets
from vbt.evaluation.posterior_source import iter_stan_source_draws, source_column_layout


class PosteriorSourceEZNTests(unittest.TestCase):
    def test_onset_recruitment_and_paper_ev(self):
        source = -np.ones((8, 3))
        source[2:, 0] = 1
        source[5:, 1] = 1
        onset, recruited = source_onsets(source)
        np.testing.assert_array_equal(onset, [2, 5, 200])
        np.testing.assert_array_equal(recruited, [True, True, False])
        raw = -np.log(((onset - onset.min()) + 1.0) / 20.0)
        expected = (raw - raw.min()) / (raw.max() - raw.min())
        np.testing.assert_allclose(epileptogenicity_value(source), expected)

    def test_constant_onset_has_no_artificial_order(self):
        source = np.ones((4, 3))
        np.testing.assert_array_equal(epileptogenicity_value(source), np.zeros(3))

    def test_stan_columns_are_reordered_to_time_region(self):
        header = ["lp__", "x.2.2", "x.1.1", "x.2.1", "x.1.2"]
        indices, n_time, n_region = source_column_layout(header)
        self.assertEqual((n_time, n_region), (2, 2))
        np.testing.assert_array_equal(indices, [2, 3, 4, 1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chain-1.csv"
            path.write_text(",".join(header) + "\n0,22,11,12,21\n", encoding="utf-8")
            draw = next(iter_stan_source_draws(path))
        np.testing.assert_array_equal(draw, [[11, 12], [21, 22]])

    def test_incomplete_stan_grid_is_rejected(self):
        with self.assertRaises(ValueError):
            source_column_layout(["x.1.1", "x.2.1", "x.1.2"])


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from vbt.models.repo_spatepi_stim import RepoSpatEpiStim7D, repo_source


class RepoSpatEpiStimTests(unittest.TestCase):
    def test_shapes_and_source_convention(self):
        model = RepoSpatEpiStim7D(np.full(3, -2.5), np.full(3, 1.8))
        state = np.repeat(
            np.array([-1.62, -16.69, 4.1, -1.11, 0.0, -0.44, 0.0])[:, None],
            3,
            axis=1,
        )
        derivative = model.derivative(state, np.zeros(3), np.zeros(3))
        self.assertEqual(derivative.shape, state.shape)
        self.assertTrue(np.isfinite(derivative).all())
        np.testing.assert_allclose(repo_source(state[None]), (state[0] - state[3])[None])

    def test_stimulus_enters_u1_and_accumulator(self):
        model = RepoSpatEpiStim7D(np.array([-2.5]), np.array([1.8]))
        state = np.array([-1.62, -16.69, 4.1, -1.11, 0.0, -0.44, 0.0])[:, None]
        zero = model.derivative(state, np.zeros(1), np.zeros(1))
        stimulated = model.derivative(state, np.zeros(1), np.ones(1))
        self.assertAlmostEqual(float(stimulated[0, 0] - zero[0, 0]), 400.0)
        self.assertAlmostEqual(float(stimulated[6, 0] - zero[6, 0]), 1000.0 / model.tau3)


if __name__ == "__main__":
    unittest.main()

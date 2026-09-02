import unittest

from vbt.profiles import PROFILES, get_profile


class ProfileTests(unittest.TestCase):
    def test_only_two_profiles(self):
        self.assertEqual(tuple(PROFILES), ("default", "vep_25"))

    def test_vep25_is_locked_to_repo_notebook_model(self):
        profile = get_profile("vep_25")
        self.assertEqual(profile.model, "repo_spatepi_stim_7d")
        self.assertEqual((profile.tt, profile.tau0, profile.tau3), (0.17, 1000.0, 600.0))
        self.assertFalse(profile.stochastic)
        self.assertEqual(profile.parcel_stimulus_scale, 0.001)

import ast,unittest
from pathlib import Path
class TestTruthIsolation(unittest.TestCase):
 def test_blind_runner_does_not_import_truth(self):
  path=Path(__file__).resolve().parents[1]/"scripts/07b_run_blind_inference.py"
  if not path.exists(): self.skipTest("Phase 7 runner not created yet")
  tree=ast.parse(path.read_text())
  text=path.read_text().lower(); self.assertNotIn("ez_truth",text); self.assertNotIn("load_ground_truth",text)
 def test_engine_exits_before_truth_and_eig_is_gain_only(self):
  path=Path(__file__).resolve().parents[1]/"src/vbt/inference/reference_engine.py"; text=path.read_text()
  self.assertLess(text.index("if args.blind_only:"),text.index("gt_path ="))
  self.assertIn("np.linalg.svd(gain",text)
  preparation=text[text.index("def main()"):text.index("if args.blind_only:")]
  self.assertNotIn("epileptor_parameters",preparation)
 def test_unlock_cannot_launch_inference(self):
  text=(Path(__file__).resolve().parents[1]/"scripts/07c_unlock_truth_and_evaluate.py").read_text().lower()
  self.assertNotIn("subprocess",text); self.assertNotIn("reference_engine",text)
 def test_source_ev_export_is_truth_free_and_unlock_cannot_infer(self):
  root=Path(__file__).resolve().parents[1]
  blind=(root/"scripts/24_export_posterior_source_ezn.py").read_text().lower()
  unlock=(root/"scripts/25_unlock_and_evaluate_source_ezn.py").read_text().lower()
  self.assertNotIn("vephypothesis",blind); self.assertNotIn("epileptor_parameters",blind)
  self.assertNotIn("subprocess",unlock); self.assertNotIn("reference_engine",unlock)
 def test_single_patient_finalize_is_blind_and_unlock_cannot_infer(self):
  root=Path(__file__).resolve().parents[1]
  blind=(root/"scripts/26_finalize_single_patient_blind.py").read_text().lower()
  unlock=(root/"scripts/27_unlock_single_patient_ezn.py").read_text().lower()
  self.assertNotIn("ez_truth",blind); self.assertNotIn("parameter_file",blind)
  self.assertNotIn("subprocess",unlock); self.assertNotIn("reference_engine",unlock)
 def test_independent_cohort_unlock_cannot_launch_inference(self):
  root=Path(__file__).resolve().parents[1]
  blind=(root/"scripts/28_run_independent_patient_cohort_blind.py").read_text().lower()
  unlock=(root/"scripts/29_unlock_independent_patient_cohort.py").read_text().lower()
  self.assertNotIn("vephypothesis",blind); self.assertNotIn("parameter_file",blind)
  self.assertNotIn("subprocess",unlock); self.assertNotIn("reference_engine",unlock)
if __name__=="__main__": unittest.main()

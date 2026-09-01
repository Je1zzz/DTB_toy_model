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
if __name__=="__main__": unittest.main()

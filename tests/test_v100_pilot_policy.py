"""Protect the existing pilot's 4-bit results when changing its arithmetic."""
import tempfile
import unittest
from pathlib import Path

from model.quant_policy import RESIDUAL_POLICY, RESIDUAL_SCHEMES, scheme_artifact_name, validate_resume_policy
from scripts import minimind_v100_pilot as pilot


class PilotPolicyTest(unittest.TestCase):
    def test_resume_accepts_current_rules_and_original_reference_formats(self):
        for scheme in RESIDUAL_SCHEMES:
            self.assertEqual(scheme_artifact_name(scheme), scheme + "_r2")
            validate_resume_policy({"quant_sim": scheme, "residual_policy": RESIDUAL_POLICY}, scheme)
            for old in ({"quant_sim": scheme}, {"quant_sim": scheme, "residual_policy": "old"}):
                with self.assertRaisesRegex(ValueError, "residual_policy"):
                    validate_resume_policy(old, scheme)
        for scheme in ("bf16", "mxfp8", "none"):
            self.assertEqual(scheme_artifact_name(scheme), scheme)
            validate_resume_policy({"quant_sim": scheme}, scheme)
        with self.assertRaisesRegex(ValueError, "quant_sim"):
            validate_resume_policy({"quant_sim": "bf16"}, "mxfp8")

    def test_pilot_only_loads_revised_3mm_weights_and_resume_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "runs/v100-pilot/data"
            data.mkdir(parents=True)
            (data / "train.jsonl").touch()
            (data / "valid.jsonl").touch()
            (root / "checkpoints").mkdir()
            (root / "out").mkdir()
            for moe in (False, True):
                arch = "_moe" if moe else ""
                legacy = f"pilot32k_s42_nvfp4_3mm_768{arch}"
                (root / "checkpoints" / f"{legacy}_resume.pth").touch()
                (root / "out" / f"{legacy}.pth").touch()
                with self.assertRaisesRegex(SystemExit, f"nvfp4_3mm_r2_768{arch}_resume"):
                    pilot.train(root, moe, "nvfp4_3mm", resume=True)
                for forward in ("same", "bf16"):
                    with self.assertRaises(FileNotFoundError) as exc:
                        pilot.evaluate(root, moe, "nvfp4_3mm", forward, "cpu")
                    self.assertEqual(exc.exception.args[0].name, f"pilot32k_s42_nvfp4_3mm_r2_768{arch}.pth")

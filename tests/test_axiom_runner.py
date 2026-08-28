import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import axiom_runner


class ResultTransportTest(unittest.TestCase):
    def test_merges_variants_and_preserves_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "cell.json"
            artifact = Path(directory) / "checkpoint.json"
            artifact.write_text("{}")
            metrics = {
                "validation_accuracy": 0.5, "trainable_parameter_count": 1,
                "wall_time_seconds": 2.0, "peak_vram_bytes": 3,
            }
            with patch.dict(os.environ, {"AXIOM_RESULT_PATH": str(result)}):
                axiom_runner.emit_result("full-finetune", 13, metrics, artifact)
                axiom_runner.emit_result("lora-r8-alpha16", 13, metrics, artifact)
            payload = json.loads(result.read_text())
            self.assertEqual(payload["schema_version"], "result.json-v1")
            self.assertEqual({row["variant"] for row in payload["results"]}, set(axiom_runner.VARIANTS))
            self.assertEqual(len(payload["artifacts"]), 2)
            self.assertTrue(payload["environment"])


if __name__ == "__main__":
    unittest.main()

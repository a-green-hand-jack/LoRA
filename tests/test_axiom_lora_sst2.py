import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

import axiom_lora_sst2 as runner


class RunnerTests(unittest.TestCase):
    def test_contract_constants(self):
        self.assertEqual(runner.VARIANTS, {"full-finetune", "lora-r8-alpha16"})
        self.assertEqual(len(runner.DATA_FILES), 3)
        self.assertEqual(runner.METRICS, (
            "validation_accuracy", "trainable_parameter_count",
            "wall_time_seconds", "peak_vram_bytes",
        ))

    def test_preflight_merges_variants_atomically(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            dataset = tmp_path / "data" / "sst2"
            dataset.mkdir(parents=True)
            original = runner.DATA_FILES.copy()
            for name in original:
                content = name.encode()
                (dataset / name).write_bytes(content)
                runner.DATA_FILES[name] = hashlib.sha256(content).hexdigest()
            result = tmp_path / "run" / "result.json"
            old_environ = os.environ.copy()
            os.environ.update(
                AXIOM_DATASET_DIR=str(tmp_path / "data"),
                AXIOM_ENVIRONMENT_DIR=str(tmp_path / "environment"),
                AXIOM_RESULT_PATH=str(result),
            )
            try:
                for variant in sorted(runner.VARIANTS):
                    sys.argv = ["runner", "--variant", variant, "--seed", "13", "--preflight"]
                    runner.main()
            finally:
                runner.DATA_FILES.clear()
                runner.DATA_FILES.update(original)
                os.environ.clear()
                os.environ.update(old_environ)
            payload = json.loads(result.read_text())
            self.assertEqual(payload["schema"], "result.json-v1")
            self.assertEqual(len(payload["results"]), 2)
            self.assertEqual(len(payload["artifacts"]), 2)
            self.assertTrue(payload["environment"])


if __name__ == "__main__":
    unittest.main()

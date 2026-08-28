import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import axiom_lora_runner as runner


class RunnerTests(unittest.TestCase):
    def test_atomic_result_merge_preserves_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            environment = {"python": "test"}
            metrics = {metric: 0 for metric in runner.METRICS}
            with mock.patch.dict(os.environ, {"AXIOM_RESULT_PATH": str(result)}):
                for variant in sorted(runner.VARIANTS):
                    row = {"variant": variant, "seed": 13, **metrics}
                    runner.atomic_merge_result(row, {"kind": "preflight", "path": variant}, environment)
            payload = json.loads(result.read_text())
            self.assertEqual(payload["transport"], "result.json-v1")
            self.assertEqual({row["variant"] for row in payload["results"]}, runner.VARIANTS)
            self.assertEqual(len(payload["artifacts"]), 2)
            self.assertEqual(payload["environment"], environment)

    def test_parse_args_requires_declared_variant_and_result(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"AXIOM_RESULT_PATH": str(Path(directory) / "out.json")}):
                args = runner.parse_args([
                    "--variant-id", "full-finetune", "--seed", "21", "--dataset-dir", directory,
                ])
            self.assertEqual(args.variant_id, "full-finetune")
            self.assertEqual(args.seed, 21)


if __name__ == "__main__":
    unittest.main()

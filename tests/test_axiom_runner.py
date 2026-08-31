import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import axiom_runner


class ResultTest(unittest.TestCase):
    def test_merge_result_preserves_other_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.json"
            first = {"variant": "full-finetune", "seed": 13, **{name: 1 for name in axiom_runner.METRICS}}
            second = {"variant": "lora-r8-alpha16", "seed": 13, **{name: 2 for name in axiom_runner.METRICS}}
            with patch.dict(os.environ, {"AXIOM_RESULT_PATH": str(target)}):
                axiom_runner.merge_result(first, Path("full.ckpt"))
                axiom_runner.merge_result(second, Path("lora.ckpt"))
            result = json.loads(target.read_text())
            self.assertEqual(result["transport"], "result.json-v1")
            self.assertEqual({row["variant"] for row in result["results"]}, {"full-finetune", "lora-r8-alpha16"})
            self.assertTrue(result["artifacts"])
            self.assertTrue(result["environment"])


if __name__ == "__main__":
    unittest.main()

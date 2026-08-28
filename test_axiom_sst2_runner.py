import json
import tempfile
import unittest
from pathlib import Path

import axiom_sst2_runner as runner


class ResultMergeTests(unittest.TestCase):
    def test_merges_variants_and_preserves_environment_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            environment = {"code_sha": "abc"}
            runner.merge_result(path, {"variant": "full-finetune", "seed": 13}, {"path": "a"}, environment)
            runner.merge_result(path, {"variant": "lora-r8-alpha16", "seed": 13}, {"path": "b"}, environment)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["schema"], "result.json-v1")
            self.assertEqual(len(payload["results"]), 2)
            self.assertEqual(payload["artifacts"], [{"path": "a"}, {"path": "b"}])
            self.assertEqual(payload["environment"], environment)


if __name__ == "__main__":
    unittest.main()

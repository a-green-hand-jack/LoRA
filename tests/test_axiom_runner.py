import json
import tempfile
import unittest
from pathlib import Path

from axiom_runner import atomic_merge_result


class AtomicMergeTest(unittest.TestCase):
    def test_preserves_other_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            environment = {"python": "test"}
            first = {"variant": "full-finetune", "seed": 13, "validation_accuracy": 0.5}
            second = {"variant": "lora-r8-alpha16", "seed": 13, "validation_accuracy": 0.6}
            atomic_merge_result(result, first, {"path": "first.pt"}, environment)
            atomic_merge_result(result, second, {"path": "second.pt"}, environment)
            payload = json.loads(result.read_text())
            self.assertEqual(payload["schema_version"], "result.json-v1")
            self.assertEqual(payload["results"], [first, second])
            self.assertEqual(payload["artifacts"], [{"path": "first.pt"}, {"path": "second.pt"}])
            self.assertEqual(payload["environment"], environment)

    def test_rejects_incompatible_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            result.write_text('{"schema_version":"other"}')
            with self.assertRaisesRegex(ValueError, "incompatible"):
                atomic_merge_result(result, {"variant": "full-finetune", "seed": 13}, {"path": "x"}, {"python": "x"})


if __name__ == "__main__":
    unittest.main()

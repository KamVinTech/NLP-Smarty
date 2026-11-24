"""Guards against re-introducing claims the measurements contradict."""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
EXT = (ROOT / "src" / "extension.ts").read_text(encoding="utf-8")


class ClaimsTest(unittest.TestCase):
    def test_readme_has_no_false_claims(self):
        for bad in ("within ~3%", "Hard ceiling enforced", "20–40 tokens per tool",
                    "Hard ceiling for entire compiled payload"):
            self.assertNotIn(bad, README)

    def test_readme_discloses_baseline_and_estimator(self):
        for needed in ("modeled baseline", "advisory", "Measured results"):
            self.assertIn(needed, README)

    def test_ui_does_not_label_estimates_as_bpe(self):
        visible = re.sub(r"t\.\w*bpe\w*", "", EXT)
        self.assertNotRegex(visible, r"\bBPE\b")


if __name__ == "__main__":
    unittest.main()

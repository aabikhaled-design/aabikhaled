import subprocess
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILTER = ROOT / "book" / "literal-tokens.lua"


def render(source, output="html"):
    return subprocess.run(
        ["pandoc", "--from", "markdown+fenced_divs", "--to", output,
         "--lua-filter", str(FILTER)],
        input=source, text=True, capture_output=True, check=True,
    ).stdout


class BookRenderingTest(unittest.TestCase):
    def test_literal_tokens_remain_text_in_valid_xhtml(self):
        source = 'Prompt: "<image A> caption <image B>". Replace a name with "<PERSON>".\n\n'
        source += '| Token | Meaning |\n| --- | --- |\n| <image> | Image |\n| <doc A> | Context |\n'
        root = ET.fromstring("<div>" + render(source) + "</div>")
        text = "".join(root.itertext())
        for token in ("<image>", "<image A>", "<image B>", "<PERSON>", "<doc A>"):
            self.assertIn(token, text)

    def test_code_and_real_html_are_unchanged(self):
        source = 'Keep `<image>` and <em>emphasis</em>.\n\n```text\n<PERSON> <doc A>\n```\n'
        result = render(source)
        root = ET.fromstring("<div>" + result + "</div>")
        self.assertEqual(root.find("p/code").text, "<image>")
        self.assertEqual(root.find("p/em").text, "emphasis")
        self.assertEqual(root.find("pre/code").text.strip(), "<PERSON> <doc A>")

    def test_pdf_input_retains_literal_tokens(self):
        result = render('"<image A>" "<PERSON>" "<doc A>"', "latex")
        for token in ("image A", "PERSON", "doc A"):
            self.assertIn(token, result)
        self.assertEqual(result.count(r"\textless"), 3)
        self.assertEqual(result.count(r"\textgreater"), 3)


if __name__ == "__main__":
    unittest.main()

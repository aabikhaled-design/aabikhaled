import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILTER = ROOT / "book" / "literal-tokens.lua"
PDF_SOURCE_FORMAT = "markdown+fenced_divs+autolink_bare_uris"
LONG_LINES = """# Wrapping regression

```python
message = "Every sample in the batch must be pre-padded with the same number of image placeholders before replacing them with projected image embeddings."
identifier = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
```

```text
This plain-text code block must also wrap its long lines without losing the final marker: PLAIN_TEXT_END.
```

Further reading: (http://neuralnetworksanddeeplearning.com/). Keep the URL clickable.

| Leaderboard | Tracks | URL |
| --- | --- | --- |
| Open ASR Leaderboard | English and multilingual | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena | English TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Escaping | Literal symbols | `{value}#100%_ok` |
"""


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

    @unittest.skipUnless(shutil.which("xelatex") and shutil.which("pdftotext"),
                         "PDF layout check requires xelatex and pdftotext")
    def test_pdf_long_code_and_urls_stay_inside_margins(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "wrapping.pdf"
            result = subprocess.run(
                ["pandoc", "--from", PDF_SOURCE_FORMAT, "--pdf-engine=xelatex",
                 "--lua-filter", str(ROOT / "book" / "pdf-layout.lua"),
                 "--include-in-header", str(ROOT / "book" / "theme.tex"),
                 "-V", "documentclass=book", "-V", "geometry=margin=1in",
                 "-o", str(pdf)],
                input=LONG_LINES, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            bbox = subprocess.check_output(["pdftotext", "-bbox", str(pdf), "-"], text=True)
        root = ET.fromstring(bbox)
        ns = {"x": "http://www.w3.org/1999/xhtml"}
        for page in root.findall(".//x:page", ns):
            right = float(page.attrib["width"]) - 72
            for word in page.findall(".//x:word", ns):
                self.assertGreaterEqual(float(word.attrib["xMin"]), 71, word.text)
                self.assertLessEqual(float(word.attrib["xMax"]), right + 1, word.text)
        text = "".join(word.text or "" for word in root.findall(".//x:word", ns))
        for marker in ("embeddings.", "PLAIN_TEXT_END.", "neuralnetworksanddeeplearning.com",
                       "open_asr_leaderboard", "TTS-Arena", "{value}#100%_ok"):
            self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()

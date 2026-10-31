"""Multi-script tokenization dispatch.

Splits text into maximal same-script runs and routes each run to its
registered ``ScriptSegmenter``. This is the only "glue" code in the
tokenization subpackage -- adding a new language later means writing one new
segmenter (see khmer_kcc.py / latin_segmenter.py for the pattern), registering
it below, and adding one Unicode-range check to ``detect_script``. No changes
are needed to existing segmenters or to the tokenizer/vocab-building code
that consumes ``MultiScriptSegmenter``.
"""

from itertools import groupby

from .base import SCRIPT_SEGMENTERS, ScriptSegmenter, detect_script, register_segmenter
from .khmer_kcc import KCCSegmenter
from .latin_segmenter import LatinCharSegmenter


class CharSegmenter:
    """Fallback segmenter: one token per character."""

    def segment(self, text: str) -> list[str]:
        return list(text)


register_segmenter("khmer", KCCSegmenter())
register_segmenter("latin", LatinCharSegmenter())
register_segmenter("digit", CharSegmenter())
register_segmenter("other", CharSegmenter())


class MultiScriptSegmenter:
    def __init__(self, segmenters: dict[str, ScriptSegmenter] | None = None):
        self.segmenters = segmenters if segmenters is not None else SCRIPT_SEGMENTERS

    def segment(self, text: str) -> list[str]:
        pieces: list[str] = []
        for script, group in groupby(text, key=detect_script):
            run = "".join(group)
            segmenter = self.segmenters[script]
            pieces.extend(segmenter.segment(run))
        return pieces

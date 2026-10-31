"""Script-dispatch tokenization interfaces.

Each script (Khmer, Latin, ...) gets its own ``ScriptSegmenter`` that knows how
to split a same-script run of text into the atomic pieces used to build the
vocabulary. Adding a new language later means writing one new segmenter and
registering it here (plus a range check in ``detect_script``) -- no changes to
existing segmenters or to the tokenizer/vocab-building code that consumes them.
"""

from typing import Protocol


class ScriptSegmenter(Protocol):
    def segment(self, text: str) -> list[str]:
        ...


KHMER_RANGE = (0x1780, 0x17FF)


def detect_script(ch: str) -> str:
    """Classify a single character into a script tag used for dispatch."""
    cp = ord(ch)
    if KHMER_RANGE[0] <= cp <= KHMER_RANGE[1]:
        return "khmer"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isascii() and ch.isdigit():
        return "digit"
    return "other"


SCRIPT_SEGMENTERS: dict[str, ScriptSegmenter] = {}


def register_segmenter(script: str, segmenter: ScriptSegmenter) -> None:
    SCRIPT_SEGMENTERS[script] = segmenter

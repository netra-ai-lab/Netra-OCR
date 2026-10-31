"""Latin-script segmentation.

Per-character for now: Latin/English has no combining-mark complexity, so
codepoint-level is already close to optimal and there is no sequence-length
problem to solve here (unlike Khmer). This is the extension point for a
future subword segmenter (e.g. a trained BPE) if English/Latin sequence
length ever becomes a bottleneck -- it would implement the same
``ScriptSegmenter`` interface without any change to the Khmer path.
"""


class LatinCharSegmenter:
    def segment(self, text: str) -> list[str]:
        return list(text)

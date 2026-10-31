"""Khmer Character Cluster (KCC) segmentation.

KCC is the established "orthographic syllable" concept from Khmer NLP research
(an inseparable character sequence, e.g. សាលាក្តី = សា+លា+ក្តី, 3 KCCs). This is a clean-room
reimplementation of the boundary rule verified against the open-source
rinabuoy/KhmerNLP word segmenter's ``is_start_of_kcc``/``seg_kcc`` (we only need
the cheap, deterministic character-class boundary rule, not their downstream
CRF/LSTM word-segmentation model, so no dependency on that package is taken).

Rule: scan left to right, accumulating characters into the current cluster.
Start a new cluster right before a character that "starts a KCC" (a consonant,
independent vowel, symbol, digit, or lunar-date character) -- unless the
current character is COENG (U+17D2), which always glues to the subscript
consonant that follows it. This single rule handles arbitrary multi-subscript
stacks (e.g. ស្ត្រី) without a hand-written repetition quantifier: COENG
never triggers a split, so the cluster keeps absorbing COENG+consonant pairs
and trailing vowel signs/diacritics until the next cluster-starting character.
"""

import re

# Consonants + independent vowels: both can start a cluster.
_CONSONANTS = "កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡ"
_INDEPENDENT_VOWELS = "អឣឤឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳ"
_KHCONST = set(_CONSONANTS + _INDEPENDENT_VOWELS)

# Dependent vowel signs + a couple of diacritics that ride along with them.
_KHVOWEL = set("ាិីឹឺុូួើឿៀេែៃោៅំះៈ")

# COENG: the subjoin marker. Never starts a cluster; always glues forward.
_KHSUB = "្"

# Remaining diacritics/signs (register shifters, robat, etc.) -- ride along.
_KHDIAC = set("៉៊់៌៍៎៏័៑៝")

# Khmer punctuation, digits, lunar-date marks: each is always its own cluster.
_KHSYM = set("។៕៖ៗ៘៙៚ " )
_KHNUMBER = set("០១២៣៤៥៦៧៨៩")
_KHLUNAR = set(chr(cp) for cp in range(0x19E0, 0x1A00))

_KHMER_BLOCK = re.compile("[ក-៿]")


def _is_khmer_char(ch: str) -> bool:
    return bool(_KHMER_BLOCK.match(ch)) or ch in _KHSYM or ch in _KHLUNAR


def _is_start_of_kcc(ch: str) -> bool:
    if _is_khmer_char(ch):
        return ch in _KHCONST or ch in _KHSYM or ch in _KHNUMBER or ch in _KHLUNAR
    return True


def segment_kcc(text: str) -> list[str]:
    """Segment a run of Khmer text into Khmer Character Clusters."""
    segs: list[str] = []
    cur = ""
    for i, c in enumerate(text):
        cur += c
        nextchar = text[i + 1] if i + 1 < len(text) else ""
        if nextchar == "" or _is_start_of_kcc(nextchar) and c != _KHSUB:
            segs.append(cur)
            cur = ""
    if cur:
        segs.append(cur)
    return segs


class KCCSegmenter:
    def segment(self, text: str) -> list[str]:
        return segment_kcc(text)

import re

# Unicode replacement char (U+FFFD) + other control junk. Latin letters and
# digits are intentionally KEPT because the vocab uses them for legitimate
# mixed Khmer text.
_NOISE_CHARS = re.compile(r"[�\x00-\x08\x0b\x0c\x0e-\x1f]")

# A multi-character cluster (2-20 chars, no spaces) immediately repeated 2+
# times within a single word, e.g. "រាំរាំ" -> "រាំ". Lazy + bounded length
# keeps it from over-collapsing or backtracking on long strings.
_REPEAT_CLUSTER = re.compile(r"(\S{2,20}?)\1{1,}")

# The same whitespace-separated token repeated consecutively, e.g.
# "H H H H" -> "H" or "អាយុ អាយុ" -> "អាយុ" (a decoder-loop artifact).
_REPEAT_TOKEN = re.compile(r"(\S+)(?:\s+\1)+")

# MRZ (machine-readable zone) chars are only A-Z, 0-9 and the '<' filler.
_MRZ_CHARS = re.compile(r"[A-Z0-9<]")


def _is_mrz_line(text: str) -> bool:
    """True for passport/ID MRZ lines, where repeated '<' filler is legitimate.

    Guards (all required) keep ordinary text that merely repeats a character from
    being mistaken for MRZ: the line must be long, contain the '<<' filler
    signature, and be almost entirely uppercase Latin / digits / chevrons.
    """
    stripped = text.replace(" ", "")
    if len(stripped) < 10 or "<<" not in stripped:
        return False
    return len(_MRZ_CHARS.findall(stripped)) / len(stripped) >= 0.85


def clean_text(text: str, max_repeat: int = 3) -> str:
    """Clean up garbage / gibberish from raw OCR decoder output.

    Steps:
        1. Strip garbage replacement / control characters.
        2. Collapse a single character repeated more than ``max_repeat`` times
           in a row (e.g. "kkkkk" -> "k").
        3. Collapse a multi-character cluster repeated within a word
           (e.g. "ប្រាំរាំរយ" -> "ប្រាំរយ").
        4. Collapse a whitespace-separated token repeated consecutively
           (e.g. "H H H H" -> "H", "អាយុ អាយុ" -> "អាយុ").
        5. Normalize whitespace left behind by removed <unk>/noise tokens.

    MRZ lines (passport/ID machine-readable zone) skip steps 2-4: their repeated
    '<' filler is legitimate, so only control/noise stripping and whitespace
    normalization are applied.
    """
    # 1. Strip garbage replacement / control characters.
    text = _NOISE_CHARS.sub("", text)
    # MRZ bypass: preserve legitimate repeated '<' filler (and any repeated MRZ
    # field chars) by skipping the repeat-collapsing steps below.
    if _is_mrz_line(text):
        return re.sub(r"\s{2,}", " ", text).strip()
    # 2. Collapse runaway repeated single-character runs.
    text = re.sub(r"(.)\1{" + str(max_repeat) + r",}", r"\1", text)
    # 3. Collapse repeated multi-character clusters inside words. Loop until
    #    stable so nested/overlapping repeats are fully reduced.
    for _ in range(3):
        collapsed = _REPEAT_CLUSTER.sub(r"\1", text)
        if collapsed == text:
            break
        text = collapsed
    # 4. Collapse consecutive duplicate whitespace-separated tokens.
    text = _REPEAT_TOKEN.sub(r"\1", text)
    # 5. Normalize whitespace.
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

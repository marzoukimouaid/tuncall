"""Arabic orthographic normalization applied to references and hypotheses before scoring (D-10).

Loanwords and dialect have no standard spelling, so variants that don't change the word
(hamza seat on alef, ta marbuta, alef maqsura, diacritics, tatweel) must not count as errors.
"""

import re
import unicodedata

_MAP = {
    **dict.fromkeys(map(ord, "أإآٱ"), "ا"),  # noqa: RUF001 (Arabic is intended)
    ord("ة"): "ه",  # noqa: RUF001
    ord("ى"): "ي",
    0x0640: None,  # tatweel
    0x0670: None,  # superscript alef
    **dict.fromkeys(range(0x064B, 0x0660), None),  # harakat, tanween, shadda, sukun, etc.
    **{0x0660 + i: str(i) for i in range(10)},  # Arabic-Indic digits
    **{0x06F0 + i: str(i) for i in range(10)},  # Extended Arabic-Indic digits
}
_SPACE = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    # NFKC folds Arabic presentation forms (U+FBxx-U+FExx) to base letters.
    text = unicodedata.normalize("NFKC", text).translate(_MAP).casefold()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return _SPACE.sub(" ", text).strip()

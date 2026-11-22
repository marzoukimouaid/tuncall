"""Latin → Arabic-script transliteration lexicon (D-10a).

``configs/data/lexicon.tsv`` holds one ``latin<TAB>arabic`` line per word. A Latin word
is a run of Latin letters (accents included, joined by apostrophes), looked up lowercased.
The same lexicon is applied to training and test text, so references and targets share
one script (D-10).
"""

import re
from pathlib import Path

_L = "[A-Za-zÀ-ÖØ-öø-ɏ]"
# French elisions stay one word: c'est, j'ai, aujourd'hui (straight or curly apostrophe).
LATIN = re.compile(f"{_L}+(?:['’]{_L}+)*")  # noqa: RUF001 (curly apostrophe is intended)


def _key(word: str) -> str:
    return word.lower().replace("\u2019", "'")  # curly and straight apostrophes share one entry


def latin_words(text: str) -> list[str]:
    return [_key(m.group()) for m in LATIN.finditer(text)]


def transliterate(text: str, lexicon: dict[str, str]) -> str:
    """Replace every Latin word found in ``lexicon``; unknown words stay as they are."""
    return LATIN.sub(lambda m: lexicon.get(_key(m.group()), m.group()), text)


def load_lexicon(path: Path) -> dict[str, str]:
    lexicon: dict[str, str] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        where = f"{path}:{n}"
        parts = line.split("\t")
        if len(parts) != 2 or not all(p.strip() for p in parts):
            raise ValueError(f"{where}: expected latin<TAB>arabic, got {line!r}")
        latin, arabic = parts
        if latin_words(latin) != [latin]:
            raise ValueError(f"{where}: key {latin!r} must be one lowercase Latin word")
        if len(arabic.split()) != 1:
            raise ValueError(f"{where}: spelling {arabic!r} must be one word (keeps word counts)")
        if LATIN.search(arabic):
            raise ValueError(f"{where}: spelling {arabic!r} still has Latin letters")
        if latin in lexicon:
            raise ValueError(f"{where}: duplicate key {latin!r}")
        lexicon[latin] = arabic
    return lexicon

from pathlib import Path

import pytest

from tuncall.data.lexicon import latin_words, load_lexicon, transliterate


def test_latin_words_are_lowercased_runs_with_accents() -> None:
    text = "و كانت شوية des Cliques في période لي فاتت 2 taxist"
    assert latin_words(text) == ["des", "cliques", "période", "taxist"]
    assert latin_words("قلت C'est bon و j\u2019ai vu l email") == [
        "c'est",
        "bon",
        "j'ai",
        "vu",
        "l",
        "email",
    ]


def test_transliterate_replaces_known_words_and_keeps_unknown() -> None:
    lexicon = {"mail": "ميل", "soi": "سوا"}
    assert transliterate("ابعث Mail لصحبتي soi-disant", lexicon) == "ابعث ميل لصحبتي سوا-disant"


def test_load_lexicon_reads_tsv(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.tsv"
    path.write_text("mail\tميل\npériode\tبيريود\n", encoding="utf-8")
    assert load_lexicon(path) == {"mail": "ميل", "période": "بيريود"}


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("mail\tميل\nmail\tميييل\n", "duplicate"),
        ("Mail\tميل\n", "lowercase Latin"),
        ("mail\tmeyl\n", "Latin letters"),
        ("sms\tاس ام اس\n", "one word"),  # noqa: RUF001
        ("mail ميل\n", "latin<TAB>arabic"),
        ("mail\t\n", "latin<TAB>arabic"),
    ],
)
def test_load_lexicon_refuses_bad_lines(tmp_path: Path, content: str, error: str) -> None:
    path = tmp_path / "lexicon.tsv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_lexicon(path)

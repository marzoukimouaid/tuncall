from latin_words import report

TEXTS = [
    ("slurp_tn", "ابعث mail لسلمى"),
    ("slurp_tn", "ابعث Mail و sms"),
    ("tedxtn", "soi disant اللي"),
    ("linto", "عسلامة"),
]


def test_report_counts_latin_tokens_left_after_the_lexicon() -> None:
    table, candidates = report(TEXTS, {"mail": "ميل"})
    assert table == (
        "| source | latin tokens | left after lexicon | unique left |\n"
        "|---|---:|---:|---:|\n"
        "| slurp_tn | 3 | 1 | 1 |\n"
        "| tedxtn | 2 | 2 | 2 |\n"
        "| linto | 0 | 0 | 0 |\n"
        "| total | 5 | 3 | 3 |"
    )
    assert candidates == [
        ("disant", 1, "tedxtn", "soi disant اللي"),
        ("sms", 1, "slurp_tn", "ابعث Mail و sms"),
        ("soi", 1, "tedxtn", "soi disant اللي"),
    ]


def test_candidates_are_sorted_by_count_then_word() -> None:
    _, candidates = report([("a", "la la le"), ("b", "le la")], {})
    assert [(w, n, s) for w, n, s, _ in candidates] == [("la", 3, "a,b"), ("le", 2, "a,b")]

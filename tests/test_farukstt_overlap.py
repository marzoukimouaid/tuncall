from farukstt_overlap import overlap

from tuncall.eval.normalize import normalize_arabic


def test_overlap_needs_four_words_and_normalizes_both_sides() -> None:
    raw_train = ["إيه باهي", "مشيت للمدرسة نهار الاثنين", "كلام آخر تماما هنا"]
    train = {normalize_arabic(t) for t in raw_train}
    refs = {
        "short": "ايه باهي",  # in train, but under 4 words: not eligible
        "seen": "مشيت للمدرسه، نهار الإثنين",  # same after normalization
        "unseen": "حاجة أخرى ما فماش منها",
    }
    eligible, seen = overlap(refs, train)
    assert eligible == ["seen", "unseen"]
    assert seen == ["seen"]

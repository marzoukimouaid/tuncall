from itertools import pairwise
from pathlib import Path

import numpy as np

from tuncall.data.conversations import (
    MAX_CLIPS,
    OVERLAP_MAX_S,
    PAUSE_S,
    assemble,
    parse,
    plan,
    read_turns,
    write_turns,
)
from tuncall.data.testsets import SAMPLE_RATE

# Four fake talks of 60 clips, 1-6 s each, times in the names like TEDxTN's.
RNG = np.random.default_rng(0)
DURATIONS = {
    f"Talk_{t}_TEDx_{i * 10000}_{i * 10000 + 5000}": round(float(RNG.uniform(1, 6)), 3)
    for t in "ABCD"
    for i in range(60)
}


def test_plan_follows_the_recipe() -> None:
    placed = plan(DURATIONS, seed=0, prefix="t")
    assert placed == plan(DURATIONS, seed=0, prefix="t")  # deterministic
    assert len({p.clip for p in placed}) == len(placed)  # no clip reused

    conversations = sorted({p.conversation for p in placed})
    assert len(conversations) > 1
    for cid in conversations:
        rows = [p for p in placed if p.conversation == cid]
        talks = {p.speaker: parse(p.clip)[0] for p in rows}
        assert sorted(talks) == ["S1", "S2"] and len(set(talks.values())) == 2
        # group rows into turns: consecutive rows of one speaker
        turns: list[list] = []
        for p in rows:
            if turns and turns[-1][0].speaker == p.speaker:
                turns[-1].append(p)
            else:
                turns.append([p])
        for turn in turns:
            assert 1 <= len(turn) <= MAX_CLIPS
            for a, b in pairwise(turn):  # clips of a turn run on in talk order, with a pause
                gap = b.start - (a.start + DURATIONS[a.clip])
                assert PAUSE_S[0] - 1e-3 <= gap <= PAUSE_S[1] + 1e-3
                assert parse(a.clip)[1] < parse(b.clip)[1]
        for prev, nxt in pairwise(turns):
            end = prev[-1].start + DURATIONS[prev[-1].clip]
            gap = nxt[0].start - end
            assert -OVERLAP_MAX_S - 1e-3 <= gap <= PAUSE_S[1] + 1e-3
            assert gap < 0 or gap >= PAUSE_S[0] - 1e-3
            assert nxt[0].start >= prev[0].start  # never starts before the turn it overlaps
        for prev, nxt in zip(turns, turns[2:], strict=False):  # same speaker, one turn apart
            assert nxt[0].start >= prev[-1].start + DURATIONS[prev[-1].clip] - 1e-3


def test_turns_tsv_round_trips(tmp_path: Path) -> None:
    placed = plan(DURATIONS, seed=1, prefix="t")
    write_turns(tmp_path / "t.tsv", placed)
    assert read_turns(tmp_path / "t.tsv") == placed


def test_assemble_places_clips_and_reference_turns() -> None:
    placed = plan(DURATIONS, seed=2, prefix="t")
    audio = {
        k: np.full(round(d * SAMPLE_RATE), 0.01 * (i + 1), dtype=np.float32)
        for i, (k, d) in enumerate(DURATIONS.items())
    }
    text = {k: k.split("_TEDx_")[1] for k in DURATIONS}
    conversations = assemble(placed, audio, text)
    assert [c.id for c in conversations] == sorted({p.conversation for p in placed})

    first = conversations[0]
    rows = [p for p in placed if p.conversation == first.id]
    # every clip's samples are in the mix at its start (plus any overlapping clip)
    expected = np.zeros_like(first.audio)
    for p in rows:
        s = round(p.start * SAMPLE_RATE)
        expected[s : s + len(audio[p.clip])] += audio[p.clip]
    np.testing.assert_array_equal(first.audio, expected)
    # reference turns cover their clips, alternate speakers and carry their texts in order
    assert first.turns[0].start == 0.0
    assert all(a.speaker != b.speaker for a, b in pairwise(first.turns))
    assert " ".join(t.text for t in first.turns) == " ".join(text[p.clip] for p in rows)
    assert max(t.end for t in first.turns) * SAMPLE_RATE == len(first.audio)
    assert [(c.speaker, c.start) for c in first.clips] == [
        (p.speaker, round(p.start * SAMPLE_RATE) / SAMPLE_RATE) for p in rows
    ]

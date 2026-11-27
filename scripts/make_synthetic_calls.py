"""Write synthetic Darija support calls with Gemini, roles marked by the writer (D-31a, D-32c).

Each call gets a domain (telecom, bank, e-commerce, in turn), a scenario and a line count
drawn with the config's seed, and one request. A reply that is not valid JSON, has a role
other than agent/customer, lacks one of the two roles, has a line count outside the range or
a line not in Arabic script is asked again (up to ``max_attempts``). Replies are cached, so a
run stopped by the daily quota resumes. When all calls exist, writes
configs/data/synthetic-calls.jsonl, the split in configs/data/synthetic-calls.yaml (with
sha256), and 20 calls for the owner's spot-check to outputs/phase5/.

    uv run python scripts/make_synthetic_calls.py [--limit N]
"""

import argparse
import random
import sys
from pathlib import Path

import yaml

from tuncall.config import load_config
from tuncall.data.testsets import REPO_ROOT, sha256_file
from tuncall.nlp.data import ROLES, Call, CallLine, CallSet, DataGen, file_sha256, is_arabic
from tuncall.nlp.llm import LLM, QuotaExhausted, parse_json

CALLS = REPO_ROOT / "configs/data/synthetic-calls.jsonl"
SCENARIOS = {
    "telecom": [
        "mobile internet is very slow",
        "a recharge was paid but the balance did not change",
        "wants to switch to a different monthly plan",
        "home fiber/ADSL line is down",
        "the bill is higher than usual",
        "lost the SIM card and wants a replacement",
        "roaming does not work abroad",
        "complains about unwanted promotional SMS",
    ],
    "bank": [
        "the card was swallowed by an ATM",
        "a transfer has not arrived",
        "an unknown charge on the account",
        "wants to open a savings account",
        "blocked card after wrong PIN entries",
        "asks about a personal loan",
        "mobile banking app login fails",
        "wants a new chequebook",
    ],
    "e-commerce": [
        "the order is late",
        "received the wrong size and wants an exchange",
        "the item arrived broken",
        "wants to cancel an order",
        "paid on delivery but has no receipt",
        "a promo code does not work",
        "asks for a refund that has not come",
        "the delivery person never called",
    ],
}
PROMPT = """Write a realistic phone call to a Tunisian customer-support line, in Tunisian \
Darija as people really speak it (not Modern Standard Arabic).
- Company type: {domain}. The customer's problem: {scenario}.
- Two speakers: the agent (the company's employee) and the customer.
- Exactly {lines} lines. One line = one speaker's turn, a short spoken sentence or two.
- Everything in Arabic script, including French and English words, written as a Tunisian \
pronounces them (مايل, ريشارج, كونيكسيون).
- Invent names, numbers and amounts; no real people or companies.
Reply with only a JSON object: {{"lines": [{{"role": "agent" or "customer", "text": "..."}}]}}"""


def valid(reply: object, lo: int, hi: int) -> list[CallLine] | None:
    if not isinstance(reply, dict) or not isinstance(reply.get("lines"), list):
        return None
    try:
        lines = [CallLine.model_validate(x) for x in reply["lines"]]
    except ValueError:
        return None
    lines = [CallLine(role=x.role, text=" ".join(x.text.split())) for x in lines]
    ok = lo <= len(lines) <= hi and {x.role for x in lines} == set(ROLES)
    return lines if ok and all(x.text and is_arabic(x.text) for x in lines) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/phase5/data.yaml")
    parser.add_argument("--limit", type=int, help="write only the first N calls")
    args = parser.parse_args()
    cfg = load_config(args.config, DataGen)
    gen = cfg.calls
    llm = LLM(cfg.provider, cfg.model, REPO_ROOT / cfg.cache_dir)

    rng = random.Random(cfg.seed)
    domains = list(SCENARIOS)
    plans = []
    for n in range(gen.count):
        domain = domains[n % len(domains)]
        plans.append(
            (domain, rng.choice(SCENARIOS[domain]), rng.randint(gen.min_lines, gen.max_lines))
        )
    calls: list[Call] = []
    try:
        for n, (domain, scenario, count) in enumerate(plans[: args.limit]):
            prompt = PROMPT.format(domain=domain, scenario=scenario, lines=count)
            for attempt in range(gen.max_attempts):
                note = f"\n(Attempt {attempt + 1}.)" if attempt else ""
                reply = llm.chat([{"role": "user", "content": prompt + note}])
                if lines := valid(parse_json(reply), gen.min_lines, gen.max_lines):
                    calls.append(
                        Call(id=f"call-{n:03d}", domain=domain, scenario=scenario, lines=lines)
                    )
                    break
            else:
                print(
                    f"\ncall-{n:03d}: no valid reply in {gen.max_attempts} tries", file=sys.stderr
                )
            print(f"\r{len(calls)} calls", end="", file=sys.stderr)
    except QuotaExhausted as err:
        print(f"\nDaily quota used up ({err}). Rerun tomorrow to resume.", file=sys.stderr)
        sys.exit(1)
    print(file=sys.stderr)

    CALLS.write_text("".join(c.model_dump_json() + "\n" for c in calls), encoding="utf-8")
    # Split planned ids, not successful ones, so a failed call never moves the others.
    ids = [f"call-{n:03d}" for n in range(gen.count)]
    random.Random(cfg.seed).shuffle(ids)
    made = {c.id for c in calls}
    splits, start = {}, 0
    for name, size in gen.splits.items():
        splits[name] = sorted(i for i in ids[start : start + size] if i in made)
        start += size
    manifest = CallSet(
        model=cfg.model,
        prompt_sha256=file_sha256(PROMPT),
        seed=cfg.seed,
        calls=str(CALLS.relative_to(REPO_ROOT)),
        calls_sha256=sha256_file(CALLS),
        splits=splits,
    )
    CALLS.with_suffix(".yaml").write_text(
        "# Generated by scripts/make_synthetic_calls.py. Do not edit; test is frozen (D-31a).\n"
        + yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    print(f"{len(calls)} calls -> {CALLS}; split { ({k: len(v) for k, v in splits.items()}) }")

    spot = REPO_ROOT / "outputs/phase5/spot-check-calls.md"
    spot.parent.mkdir(parents=True, exist_ok=True)
    out = ["# Synthetic support calls: spot-check sample\n"]
    for c in random.Random(cfg.seed).sample(calls, min(20, len(calls))):
        out.append(f"## {c.id} ({c.domain}: {c.scenario})\n")
        out += [f"- **{x.role}:** {x.text}" for x in c.lines]
        out.append("")
    spot.write_text("\n".join(out), encoding="utf-8")
    print(f"spot-check sample -> {spot}")


if __name__ == "__main__":
    main()

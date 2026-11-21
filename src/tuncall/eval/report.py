"""Rewrite generated sections of a Markdown file between ``<!-- name:start/end -->`` markers."""

import re


def replace_block(text: str, name: str, body: str) -> str:
    """Return ``text`` with the content between the ``name`` markers replaced by ``body``."""
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if len(pattern.findall(text)) != 1:
        raise ValueError(f"expected exactly one {start} ... {end} block")
    return pattern.sub(lambda _: f"{start}\n{body.strip()}\n{end}", text)

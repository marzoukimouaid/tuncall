# tuncall

Speech and language intelligence for **Tunisian Darija** customer-support calls: audio in, then diarized transcripts, intents, sentiment, entities, summaries, search and analytics out.

**Status:** Phase 1 (zero-shot baselines on a frozen test set). There are no results yet. Every number that appears here will link to the eval script and config that produced it.

The project plan, architecture and decision log are in [PROJECT.md](PROJECT.md).

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run pre-commit install
uv run pytest
```

## License

Code: [MIT](LICENSE). Dataset and model licenses will be stated on their own cards.

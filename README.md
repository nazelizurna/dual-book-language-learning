# Bilingual Book Builder

Turn two EPUB editions of the same book into one bilingual EPUB for language learning, using the [Ilya Frank method](https://en.wikipedia.org/wiki/Ilya_Frank): the original text with the translation inline, followed by a clean copy of the original for re-reading.

Sentences are matched across languages with [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) embeddings and an order-preserving alignment, so the two books stay in sync from start to finish.

## Versions

| Version | Path | Status |
|---|---|---|
| Command-line script | [`versions/cli`](versions/cli) | Working |

More versions will be added under `versions/`. Each one is self-contained.

## Quick start (CLI)

```bash
pip install -r requirements.txt

python versions/cli/frank_epub.py french.epub russian.epub out.epub \
    --orig-lang fr --trans-lang ru
```

Or put `english.epub`, `french.epub` and `russian.epub` in the working directory and run with no arguments. It builds every available pair (fr→ru, en→ru, fr→en).

## Output layout

For each group of 5 sentences:

> Original sentence one. *(Translation of sentence one.)* Original sentence two. *(Translation of sentence two.)* …
>
> │ Original sentences again, clean, for re-reading.

## Limitations

- Both editions must follow the same text. Long extra sections in one of them (prefaces, notes) can shift the alignment. Tune `band_frac` in `align()` if so.
- Sentence splitting is rule-based and tuned for English, French and Russian.
- The first run downloads the LaBSE model (~1.8 GB).
- Use only books you have the right to copy.

## Contributing

Issues and pull requests are welcome. New app versions go in their own folder under `versions/`.

## License

MIT. See [LICENSE](LICENSE).

# CLI version

Single-file command-line tool: `frank_epub.py`.

```bash
python frank_epub.py <original.epub> <translation.epub> [output.epub] \
    --orig-lang fr --trans-lang ru
```

Arguments:

- `orig`, `trans`: the two EPUB files (original first)
- `out`: output path (default `<orig-lang>_<trans-lang>_frank.epub`)
- `--orig-lang`, `--trans-lang`: language codes, default `en` and `ru`

With no arguments it looks for `english.epub`, `french.epub` and `russian.epub` in the current directory.


# LLM versiom

Turns two EPUBs of the same book (the original and a translation) into a **Ilya Frank-style reading EPUB**: the original text with short green bracketed comments after each fragment.

```
I have seldom heard him mention her (я редко слышал, /чтобы/ он называл ее; to hear — слышать; to mention — называть, упоминать; heard him mention — сложное дополнение)
```

Each bracket holds a literal translation, word glosses and a short grammar or style note. The amount of help depends on your level (A2, B1 or B2).

## How it works

1. Extracts text from both EPUBs and splits it into sentences.
2. Aligns the sentences of the two books with [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) embeddings and order-preserving dynamic programming.
3. Sends the aligned pairs to an LLM (Groq or a local Ollama model), which cuts each sentence into fragments and writes the bracket contents.
4. Writes a new EPUB. Each block shows the annotated text, an optional IPA line for rare English words, and the clean original text.

Answers are cached in `<output>.cache.json`, so an interrupted run resumes where it stopped.

## Install

```bash
pip install numpy requests beautifulsoup4 ebooklib sentence-transformers
pip install wordfreq pronouncing   # optional: pronunciation hints (English only)
```

You also need one of:
- a free [Groq API key](https://console.groq.com/keys), or
- [Ollama](https://ollama.com/download) with a model pulled, e.g. `ollama pull qwen2.5:7b`.

## Usage

```bash
python frank_epub.py original.epub translation.epub output.epub \
    --orig-lang en --trans-lang ru --level B1 --provider groq
```

If you omit the level or provider, the script asks. Without file arguments it looks for `english.epub`, `french.epub` and `russian.epub` in the current folder.

| Option | Meaning |
|---|---|
| `--level A2\|B1\|B2` | Reader level: smaller fragments and more help at A2 |
| `--provider groq\|ollama` | LLM service |
| `--model NAME` | Force a specific model (otherwise chosen automatically with fallbacks) |
| `--limit N` | Process only the first N units, useful for a test run |
| `--batch N`, `--workers N` | Sentences per request and parallel requests |
| `--group N` | Sentences per block in the output |
| `--no-ipa` | Disable the pronunciation line |
| `--debug-align FILE.tsv` | Dump the sentence alignment for checking |
| `--reset-key` | Forget the saved API key and ask again |

Try `--limit 30` first to check the result and the alignment before processing a whole book.

## API key

The key is read from the `GROQ_API_KEY` environment variable, or from `~/.frank_epub_keys.json`. If neither exists, you are asked once and the key is saved there (owner-only permissions where supported).

## Notes

- Both EPUBs must be the same book. A different edition or missing chapters can reduce alignment quality, so check with `--debug-align`.
- Quality depends on the model. Larger models give better comments.
- Use only books you have the right to copy and modify.

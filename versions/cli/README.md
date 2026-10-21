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

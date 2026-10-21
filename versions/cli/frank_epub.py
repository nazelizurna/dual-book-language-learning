import argparse
import html
import os
import re
import warnings

import numpy as np
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading multilingual LaBSE model...")
        _model = SentenceTransformer("sentence-transformers/LaBSE")
    return _model

BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]


def extract_blocks_from_epub(epub_path):
    book = epub.read_epub(epub_path)
    blocks = []

    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        if "nav" in (getattr(item, "properties", None) or []):
            continue

        soup = BeautifulSoup(item.get_content(), "html.parser")
        for el in soup(["script", "style", "head", "nav"]):
            el.decompose()

        found = soup.find_all(BLOCK_TAGS)
        if found:
            for el in found:
                text = " ".join(el.get_text(" ").split())
                if text:
                    blocks.append(text)
        else:
            text = " ".join(soup.get_text(" ").split())
            if text:
                blocks.append(text)
    return blocks

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "prof", "mme", "mlle", "mm", "me", "no",
    "vs", "etc", "cf", "p", "pp", "vol", "ch", "sr", "jr",
    "см", "ср", "т", "д", "г", "гг", "ул", "им", "проф", "т.д", "т.п", "т.е",
    "т.к", "напр", "стр", "рис", "гл", "св",
}
SENT_END = re.compile(r"[.!?…]+[\"'»”’)\]]*\s+")
OPENERS = "«\"“‘'(—–- "


def split_sentences(text):
    text = " ".join(text.split())
    sentences, pos = [], 0

    for m in SENT_END.finditer(text):
        end = m.end()
        marker = m.group().strip()

        if marker.startswith(".") and not marker.startswith(".."):
            words = text[pos:m.start()].split()
            if words:
                token = words[-1].strip("«\"“‘'(").lower()
                if token in ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
                    continue

        nxt = text[end:end + 4].lstrip(OPENERS)
        if nxt and not (nxt[0].isupper() or nxt[0].isdigit()):
            continue

        sentence = text[pos:end].strip()
        if sentence:
            sentences.append(sentence)
        pos = end

    tail = text[pos:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def blocks_to_sentences(blocks):
    sentences = []
    for b in blocks:
        sentences.extend(split_sentences(b))
    return sentences


def _normalize(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


def _pair_embeddings(E):
    P = np.zeros_like(E)
    P[1:] = E[:-1] + E[1:]
    return _normalize(P)


def align(O, T, threshold=0.35, gap=0.25, band_frac=0.06, min_band=60):
    N, M = len(O), len(T)
    hw = max(min_band, int(band_frac * max(N, M)))
    O2, T2 = _pair_embeddings(O), _pair_embeddings(T)

    NEG = -1e18
    MOVES = {1: (1, 1), 2: (2, 1), 3: (1, 2), 4: (1, 0), 5: (0, 1)}
    los = [0] * (N + 1)
    score = [None] * (N + 1)
    back = [None] * (N + 1)

    def get(r, j):
        k = j - los[r]
        row = score[r]
        return row[k] if 0 <= k < len(row) else NEG

    for i in range(N + 1):
        c = int(round(i * M / max(N, 1)))
        lo, hi = max(0, c - hw), min(M, c + hw)
        los[i] = lo
        sc = [NEG] * (hi - lo + 1)
        bk = [0] * (hi - lo + 1)
        score[i], back[i] = sc, bk

        a = max(lo, 1)
        if i >= 1 and a <= hi:
            Tseg = T[a - 1:hi]
            s11 = Tseg @ O[i - 1]
            s12 = T2[a - 1:hi] @ O[i - 1]
            s21 = Tseg @ O2[i - 1] if i >= 2 else None

        for j in range(lo, hi + 1):
            best, mv = NEG, 0
            if i == 0 and j == 0:
                best = 0.0
            if i >= 1 and j >= 1:
                p = get(i - 1, j - 1)
                if p > NEG:
                    v = p + (float(s11[j - a]) - threshold)
                    if v > best:
                        best, mv = v, 1
            if i >= 2 and j >= 1:
                p = get(i - 2, j - 1)
                if p > NEG:
                    v = p + (float(s21[j - a]) - threshold) * 1.5 - 0.05
                    if v > best:
                        best, mv = v, 2
            if i >= 1 and j >= 2:
                p = get(i - 1, j - 2)
                if p > NEG:
                    v = p + (float(s12[j - a]) - threshold) * 1.5 - 0.05
                    if v > best:
                        best, mv = v, 3
            if i >= 1:
                p = get(i - 1, j)
                if p > NEG and p - gap > best:
                    best, mv = p - gap, 4
            if j > lo:
                p = sc[j - 1 - lo]
                if p > NEG and p - gap > best:
                    best, mv = p - gap, 5
            sc[j - lo] = best
            bk[j - lo] = mv

    pairs, i, j = [], N, M
    while i > 0 or j > 0:
        mv = back[i][j - los[i]]
        if mv == 0:
            break
        di, dj = MOVES[mv]
        pairs.append((i - di, i, j - dj, j))
        i, j = i - di, j - dj
    pairs.reverse()
    return pairs

ARPA2IPA = {
    "AA": "ɑː", "AE": "æ", "AH": "ʌ", "AO": "ɔː", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "e", "ER": "ɜː",
    "EY": "eɪ", "F": "f", "G": "g", "HH": "h", "IH": "ɪ", "IY": "iː",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "əʊ", "OY": "ɔɪ", "P": "p", "R": "r", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "uː", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}


def make_ipa_hinter(max_words=3, zipf_max=3.4, min_len=5):
    try:
        import pronouncing
        from wordfreq import zipf_frequency
    except ImportError:
        print("  (pronunciation hints skipped: pip install wordfreq pronouncing)")
        return None

    def to_ipa(arpa):
        out = []
        for ph in arpa.split():
            base, stress = ph.rstrip("012"), ph[-1] if ph[-1] in "012" else ""
            sym = ARPA2IPA.get(base, "")
            if stress == "1":
                out.append("\u02c8")
            elif stress == "2":
                out.append("\u02cc")
            out.append(sym)
        return "".join(out)

    def hint(text):
        seen, cands = set(), []
        for w in re.findall(r"[A-Za-z']+", text):
            lw = w.lower().strip("'")
            if lw in seen or len(lw) < min_len:
                continue
            seen.add(lw)
            if zipf_frequency(lw, "en") <= zipf_max:
                ph = pronouncing.phones_for_word(lw)
                if ph:
                    cands.append((zipf_frequency(lw, "en"), lw, to_ipa(ph[0])))
        cands.sort()
        return ", ".join(f"{w} /{i}/" for _, w, i in cands[:max_words])

    return hint

_TRAIL = re.compile(r"([.!?\u2026]+[\"'\u00bb\u201d\u2019)\]]*)$")

_QUOTE_PAIRS = [("\u201c", "\u201d"), ("\u00ab", "\u00bb"), ("\u2018", "\u2019")]


def strip_stray_quotes(t):
    for op, cl in _QUOTE_PAIRS[:2]:
        if t.count(op) != t.count(cl):
            t = t.replace(op, "").replace(cl, "")
    if t.count('"') % 2:
        t = t.replace('"', "")
    return t.strip()


def _mix(o, t):
    if not t:
        return html.escape(o)
    m = _TRAIL.search(o)
    body, tail = (o[:m.start()], m.group(1)) if m else (o, "")
    t = strip_stray_quotes(t).rstrip(".")
    return (f'{html.escape(body.rstrip())} <span class="tr" style="color:#3b8a5a">'
            f'({html.escape(t)})</span>{html.escape(tail)}')


def build_epub(pairs, output_path, orig_lang, trans_lang, title, group=3, ipa=None):
    book = epub.EpubBook()
    book.set_identifier(f"frank_{orig_lang}_{trans_lang}_{abs(hash(title)) % 10**8}")
    book.set_title(title)
    book.set_language(orig_lang)

    css = epub.EpubItem(
        uid="style", file_name="style.css", media_type="text/css",
        content=(
            "body{font-family:serif;line-height:1.6;}"
            "div.blk{background:#f8f9fb;border-left:4px solid #dfe1ea;"
            "padding:.6em 1em;margin:0 0 1.4em 0;}"
            "div.blk p{margin:0 0 .9em 0;}"
            "div.blk p:last-child{margin-bottom:0;}"
            "span.tr{color:#3b8a5a;}"
            "p.ipa{color:#000;}"
        ),
    )
    book.add_item(css)

    PER_FILE = 90
    chapters = []

    for start in range(0, len(pairs), PER_FILE):
        part = pairs[start:start + PER_FILE]
        body = []
        for g in range(0, len(part), group):
            grp = part[g:g + group]
            mixed = " ".join(_mix(o, t) for o, t in grp)
            clean_text = " ".join(o for o, _ in grp)
            block = [f'<p class="mix">{mixed}</p>']
            if ipa:
                h = ipa(clean_text)
                if h:
                    block.append(f'<p class="ipa">{html.escape(h)}</p>')
            block.append(f'<p class="clean">{html.escape(clean_text)}</p>')
            body.append('<div class="blk">' + "".join(block) + "</div>")

        n = len(chapters) + 1
        ch = epub.EpubHtml(title=f"Part {n}", file_name=f"part_{n}.xhtml", lang=orig_lang)
        ch.content = f"<h2>Part {n}</h2>" + "".join(body)
        ch.add_item(css)
        book.add_item(ch)
        chapters.append(ch)

    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + chapters
    epub.write_epub(output_path, book)


def create_frank_epub(orig_epub, trans_epub, output_epub, orig_lang="en", trans_lang="ru",
                      group=3, use_ipa=True, debug_tsv=None):
    print(f"\nAligning [{orig_lang.upper()}] {orig_epub}  ->  [{trans_lang.upper()}] {trans_epub}")

    print("Step 1: extracting text...")
    orig_sents = blocks_to_sentences(extract_blocks_from_epub(orig_epub))
    trans_sents = blocks_to_sentences(extract_blocks_from_epub(trans_epub))
    print(f"  original: {len(orig_sents)} sentences, translation: {len(trans_sents)} sentences")
    if not orig_sents or not trans_sents:
        print("Error: one of the books produced no text.")
        return

    print("Step 2: computing embeddings...")
    model = get_model()
    O = model.encode(orig_sents, batch_size=64, show_progress_bar=True,
                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    T = model.encode(trans_sents, batch_size=64, show_progress_bar=True,
                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    print("Step 3: aligning (order-preserving)...")
    spans = align(O, T)

    if debug_tsv:
        with open(debug_tsv, "w", encoding="utf-8") as fh:
            fh.write("sim\toriginal\ttranslation\n")
            for i0, i1, j0, j1 in spans:
                sim = ""
                if i1 > i0 and j1 > j0:
                    a, b = O[i0:i1].mean(0), T[j0:j1].mean(0)
                    sim = f"{float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))):.2f}"
                fh.write(f"{sim}\t{' '.join(orig_sents[i0:i1])}\t{' '.join(trans_sents[j0:j1])}\n")
        print(f"  alignment dump: {debug_tsv}")

    pairs = []
    for i0, i1, j0, j1 in spans:
        o = " ".join(orig_sents[i0:i1])
        t = " ".join(trans_sents[j0:j1])
        if o:
            pairs.append((o, t))
    unmatched = sum(1 for _, t in pairs if not t)
    print(f"  {len(pairs)} aligned units ({unmatched} without translation)")

    print("Step 4: writing EPUB...")
    ipa = make_ipa_hinter() if (use_ipa and orig_lang == "en") else None
    build_epub(pairs, output_epub, orig_lang, trans_lang,
               f"Bilingual Book ({orig_lang.upper()} - {trans_lang.upper()})",
               group=group, ipa=ipa)
    print(f"Done: {output_epub}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orig", nargs="?")
    ap.add_argument("trans", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--orig-lang", default="en")
    ap.add_argument("--trans-lang", default="ru")
    ap.add_argument("--group", type=int, default=3, help="sentences per block (default 3)")
    ap.add_argument("--debug-align", metavar="FILE.tsv", help="write the sentence alignment (with similarity) to a TSV to inspect it")
    ap.add_argument("--no-ipa", action="store_true", help="disable pronunciation line (English only)")
    args = ap.parse_args()

    if args.orig and args.trans:
        out = args.out or f"{args.orig_lang}_{args.trans_lang}_frank.epub"
        create_frank_epub(args.orig, args.trans, out, args.orig_lang, args.trans_lang,
                          args.group, not args.no_ipa, args.debug_align)
        return

    have = {l: os.path.exists(f) for l, f in
            {"en": "english.epub", "fr": "french.epub", "ru": "russian.epub"}.items()}
    names = {"en": "english.epub", "fr": "french.epub", "ru": "russian.epub"}
    done = False
    for o, t in (("fr", "ru"), ("en", "ru"), ("fr", "en")):
        if have[o] and have[t]:
            create_frank_epub(names[o], names[t], f"{names[o][:-5]}_{names[t][:-5]}_frank.epub", o, t,
                              args.group, not args.no_ipa, args.debug_align)
            done = True
    if not done:
        print("Could not find a valid pair. Name files english.epub / french.epub / russian.epub "
              "or pass paths explicitly.")

if __name__ == "__main__":
    main()

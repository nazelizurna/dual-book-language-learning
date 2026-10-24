import argparse
import getpass
import hashlib
import html
import json
import os
import re
import sys
import threading
import time
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_model = None

PROMPT_VERSION = "v4-languages"


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


def align(O, T, threshold=0.35, gap=0.25, band_frac=0.15, min_band=250):
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


_TRAIL = re.compile(r"([.!?…]+[\"'»”’)\]]*)$")
_TAIL = re.compile(r"([\s.,;:!?…\"'»”’)\]–—-]+)$")
_QUOTE_PAIRS = [("“", "”"), ("«", "»")]
GREEN = "#3b8a5a"

PROVIDERS = {
    "groq": {
        "label": "Groq (online)", "base": "https://api.groq.com/openai/v1", "env": "GROQ_API_KEY",
        "default": "llama-3.3-70b-versatile", "key_url": "https://console.groq.com/keys",
        "note": "free tier with rate limits; fast; needs a free API key",
    },
    "ollama": {
        "label": "Ollama (local)", "base": "http://localhost:11434/v1", "env": None,
        "default": None, "key_url": "https://ollama.com/download",
        "note": "runs on your computer, no key, no internet needed",
    },
}

SKIP_MODEL_WORDS = ("embed", "whisper", "tts", "guard", "moderation", "image", "imagen", "veo",
                    "audio", "transcribe", "speech", "dall", "aqa", "rerank", "orpheus", "playai", "allam", "compound", "safeguard")

LANG_NAMES = {
    "en": "English", "fr": "French", "ru": "Russian", "de": "German",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "pl": "Polish",
}

LEVELS = {
    "A2": {
        "desc": "elementary",
        "fragments": "very short fragments of 2 to 5 words",
        "gloss": "every word or phrase beyond the 1000 most common words, and all idioms, collocations and fixed expressions",
        "max_words": 4,
        "note": "give a note for almost every sentence that has any construction beyond the basic present, past and future",
    },
    "B1": {
        "desc": "intermediate",
        "fragments": "short fragments of 3 to 7 words",
        "gloss": "words beyond roughly the 3000 most common words, idioms, collocations, fixed expressions and figurative uses",
        "max_words": 3,
        "note": "give a note only for non-trivial constructions",
    },
    "B2": {
        "desc": "upper-intermediate",
        "fragments": "longer fragments, up to a full clause",
        "gloss": "only rare or literary words, idioms and figurative uses; skip anything an upper-intermediate reader knows",
        "max_words": 3,
        "note": "give a note only for genuinely tricky grammar or a notable stylistic or idiomatic point",
    },
}

GRAMMAR_TOPICS = {
    "en": "perfect tenses, passive, modal verbs, participle clauses, complex object, inversion, conditionals, reported speech, phrasal verbs, tricky prepositions",
    "fr": "passe simple and imparfait, subjonctif, pronouns (y, en, order of le/la/les), agreement of the past participle, negation (ne ... pas/jamais/que), gerondif, infinitive constructions, literary turns",
    "ru": "verbal aspect, cases and prepositions, participles and gerunds, verbs of motion, reflexive verbs, clauses with chtoby, word order, impersonal sentences",
    "de": "case and gender, word order in subordinate clauses, separable verbs, Konjunktiv, passive, modal particles",
    "es": "subjuntivo, ser and estar, preterito and imperfecto, clitic pronouns, por and para",
    "it": "congiuntivo, passato remoto, clitic pronouns, partitive, prepositions",
}

STOPWORDS = {
    "en": set("the and of to a in that is was he she it for with as his her on be at by had not but they this from or have an are you i".split()),
    "fr": set("le la les de des du un une et est que qui dans pour pas sur il elle au avec ce se ne son sa ses plus par mais on nous vous je tu ils elles etait ete".split()),
    "de": set("der die das und ist nicht ein eine zu den mit von sich auf fur als auch es dem ich er sie war".split()),
    "es": set("el la los las de que y en un una es por con no para su se al lo como mas pero".split()),
    "it": set("il lo la i gli le di che e un una per non con si da come piu ma".split()),
}


def guess_language(sentences):
    step = max(1, len(sentences) // 400)
    text = " ".join(sentences[::step][:400])
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None, {}
    cyr = sum(1 for c in letters if "\u0400" <= c <= "\u04ff")
    if cyr / len(letters) > 0.5:
        return "ru", {"ru": 1.0}
    tokens = re.findall(r"[^\W\d_]+", unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().casefold())
    if not tokens:
        return None, {}
    scores = {l: sum(1 for t in tokens if t in sw) / len(tokens) for l, sw in STOPWORDS.items()}
    return max(scores, key=scores.get), scores


def check_language(sentences, expected, role):
    guess, scores = guess_language(sentences)
    if guess is None or guess == expected:
        return
    if expected != "ru" and expected not in scores:
        return
    if expected != "ru" and scores.get(guess, 0) < 1.3 * scores.get(expected, 0):
        return
    print(f"\nWARNING: the {role} EPUB looks like {LANG_NAMES.get(guess, guess)}, but you chose "
          f"{LANG_NAMES.get(expected, expected)}. Mixing up the files gives wrong comments.")
    try:
        a = input("Continue anyway? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        a = "y"
    if a not in ("y", "yes"):
        sys.exit("Stopped. Check which EPUB is which language and run it again.")


def ask(prompt, secret=False):
    try:
        if secret and sys.stdin.isatty():
            return getpass.getpass(prompt).strip()
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def choose_language(question, exclude=None):
    options = [c for c in ("en", "fr", "ru") if c != exclude]
    print(question)
    for k, c in enumerate(options, 1):
        print(f"  {k}) {LANG_NAMES[c]}")
    while True:
        a = ask(f"Choose 1-{len(options)} or type a language code [1]: ").lower()
        if not a:
            return options[0]
        if a.isdigit() and 1 <= int(a) <= len(options):
            return options[int(a) - 1]
        if a in LANG_NAMES and a != exclude:
            return a
        print("Please choose one of the listed languages.")


def find_book(lang, role):
    name = LANG_NAMES.get(lang, lang).lower()
    for cand in (f"{name}.epub", f"{lang}.epub"):
        if os.path.exists(cand):
            print(f"  {role}: using {cand}")
            return cand
    for _ in range(3):
        path = ask(f"Path to the {LANG_NAMES.get(lang, lang)} EPUB ({role}): ")
        path = os.path.expanduser(path.strip().strip("\"'").replace("\\ ", " "))
        if path and os.path.exists(path):
            return path
        print("File not found.")
    sys.exit("No EPUB file given.")


def choose_level():
    options = ["A2", "B1", "B2"]
    print("Reader level:")
    for k, o in enumerate(options, 1):
        print(f"  {k}) {o}")
    while True:
        a = ask("Choose 1-3 or type the level [B1]: ").upper()
        if not a:
            return "B1"
        if a in options:
            return a
        if a.isdigit() and 1 <= int(a) <= 3:
            return options[int(a) - 1]
        print("Please choose A2, B1 or B2.")


def choose_provider():
    names = list(PROVIDERS)
    print("Service:")
    for k, n in enumerate(names, 1):
        print(f"  {k}) {PROVIDERS[n]['label']} - {PROVIDERS[n]['note']}")
    while True:
        a = ask("Choose 1-%d or type the name [1]: " % len(names)).lower()
        if not a:
            return names[0]
        if a in PROVIDERS:
            return a
        if a.isdigit() and 1 <= int(a) <= len(names):
            return names[int(a) - 1]
        print("Please choose one of the listed services.")


def get_api_key(prov):
    if prov["env"] is None:
        return "none"
    print(f"Get a free {prov['label']} key at {prov['key_url']}")
    while True:
        key = ask(f"{prov['label']} API key: ", secret=True)
        if key:
            return key
        print("The key can't be empty.")


def _frag(src, tr, words, note=""):
    return {"src": src, "tr": tr, "words": [{"w": w, "tr": m} for w, m in words], "note": note}


_EN_SRC = ("To Sherlock Holmes she is always THE woman. I have seldom heard him mention her "
           "under any other name.")
_FR_SRC = "Je ne l'ai presque jamais entendu la nommer autrement que par son prénom."
_RU_SRC = "Я почти никогда не слышал, чтобы он называл ее по имени."

EXAMPLES = {
    ("en", "ru"): (_EN_SRC, [
        _frag("To Sherlock Holmes she is always THE woman.",
              "для Шерлока Холмса она всегда та самая: «эта» женщина", [],
              "THE выделено заглавными: «единственная, неповторимая»"),
        _frag("I have seldom heard him mention her", "я редко слышал, /чтобы/ он называл ее",
              [("to hear", "слышать"), ("to mention", "называть, упоминать")],
              "heard him mention — сложное дополнение: слышал, как он (что-то) делает"),
        _frag("under any other name.", "/под/ каким-либо другим именем", []),
    ]),
    ("en", "fr"): (_EN_SRC, [
        _frag("To Sherlock Holmes she is always THE woman.",
              "pour Sherlock Holmes elle est toujours LA femme", [],
              "THE en majuscules: la seule, l'unique"),
        _frag("I have seldom heard him mention her", "je l'ai rarement entendu la nommer",
              [("to hear", "entendre"), ("to mention", "mentionner, nommer")],
              "heard him mention: verbe de perception + infinitif, comme en français"),
        _frag("under any other name.", "sous tout autre nom", []),
    ]),
    ("fr", "en"): (_FR_SRC, [
        _frag("Je ne l'ai presque jamais entendu", "I have hardly ever heard him",
              [("presque jamais", "hardly ever"), ("entendre", "to hear")],
              "ne ... jamais: negation wraps the verb"),
        _frag("la nommer", "call her", [("nommer", "to name, to call")],
              "entendre + infinitive: heard him call her"),
        _frag("autrement que par son prénom.", "other than by her first name",
              [("autrement que", "other than"), ("prénom", "first name")]),
    ]),
    ("fr", "ru"): (_FR_SRC, [
        _frag("Je ne l'ai presque jamais entendu", "я почти никогда не слышал, /чтобы/ он",
              [("presque jamais", "почти никогда"), ("entendre", "слышать")],
              "ne ... jamais: отрицание обрамляет глагол"),
        _frag("la nommer", "называл ее", [("nommer", "называть")],
              "entendu + инфинитив: слышал, как он называет"),
        _frag("autrement que par son prénom.", "иначе, чем /по/ имени",
              [("autrement que", "иначе чем"), ("prénom", "имя")]),
    ]),
    ("ru", "en"): (_RU_SRC, [
        _frag("Я почти никогда не слышал,", "I have hardly ever heard",
              [("почти никогда", "hardly ever"), ("слышать", "to hear")],
              "не ... никогда: double negation is normal in Russian"),
        _frag("чтобы он называл ее", "him call her", [("называть", "to call, to name")],
              "чтобы + past form: clause about what someone does"),
        _frag("по имени.", "by her first name", [("имя", "name, first name")]),
    ]),
    ("ru", "fr"): (_RU_SRC, [
        _frag("Я почти никогда не слышал,", "je n'ai presque jamais entendu",
              [("почти никогда", "presque jamais"), ("слышать", "entendre")],
              "double négation: не ... никогда"),
        _frag("чтобы он называл ее", "qu'il l'appelle", [("называть", "appeler, nommer")],
              "чтобы + passé: ici subjonctif en français"),
        _frag("по имени.", "par son prénom", [("имя", "nom, prénom")]),
    ]),
}


def _preview(frag):
    parts = [frag["tr"]] + [f"{w['w']} — {w['tr']}" for w in frag["words"]]
    if frag["note"]:
        parts.append(frag["note"])
    return f"{frag['src']} ({'; '.join(parts)})"


def system_prompt(level, orig_name, trans_name, orig_lang=None, trans_lang=None):
    cfg = LEVELS[level]
    topics = GRAMMAR_TOPICS.get(orig_lang, "")
    topics_line = f" Typical topics in {orig_name}: {topics}." if topics else ""
    verb_form = 'Use the infinitive form for verbs (with "to" in English). ' if orig_lang == "en" else "Use the infinitive form for verbs. "
    example = EXAMPLES.get((orig_lang, trans_lang))
    if example:
        ex_src, ex_frags = example
        ex_json = json.dumps({"items": [{"id": 0, "fragments": ex_frags}]}, ensure_ascii=False)
        shown = _preview(ex_frags[1])
        preview_line = f"  {shown}\n"
        example_line = f"Example. Input src: {ex_src!r}. A good reply:\n{ex_json}\n"
    else:
        preview_line = f"  <{orig_name} fragment> (<{trans_name} translation>; <word> — <meaning>; <comment>)\n"
        example_line = ""
    return (
        f"You prepare {orig_name} text for a {trans_name}-speaking learner at CEFR level {level} ({cfg['desc']}).\n"
        f"You receive a JSON list of items. Each item has: id, src (a {orig_name} sentence or passage) "
        f"and ref (an existing {trans_name} translation of it, possibly empty).\n"
        "For every item, split src into consecutive fragments, in order, covering all of src.\n"
        f"Languages: src and every \"w\" are {orig_name}. tr, every word meaning and every note are written in "
        f"{trans_name} only. Never use a third language.\n"
        "The learner will read each fragment followed by a bracket, like this:\n"
        f"{preview_line}"
        "so what you write must be a useful, compact reading aid, not a literary translation.\n"
        "Rules:\n"
        "1. Joining all fragment src values with single spaces must reproduce src exactly, character for character, "
        "punctuation included. Never change, drop, reorder or add anything. Copy the characters exactly as given.\n"
        f"2. Use {cfg['fragments']}.\n"
        f"3. tr: a close, literal-leaning {trans_name} rendering of the fragment that lets the learner map the words one to one. "
        "Stay close to the wording of ref where it fits. "
        f"Put words that {trans_name} grammar requires but the {orig_name} fragment lacks between slashes, like /word/. "
        "Do not end tr with a comma, full stop or any punctuation. tr must never be empty.\n"
        f"4. words: a list of objects {{\"w\": dictionary form of a word or phrase from the fragment, \"tr\": its short {trans_name} meaning in this context}}. "
        f"Include: {cfg['gloss']}. {verb_form}"
        f"At most {cfg['max_words']} per fragment, choose the most useful ones. Use an empty list only if the fragment truly has nothing worth glossing.\n"
        f"5. note: an empty string, or one very short comment in {trans_name} (at most 15 words, no full stop at the end) "
        f"about grammar, an idiom, a nuance or style. At this level, {cfg['note']}.{topics_line} "
        "Name the construction and show it with the words from the fragment, do not just repeat the translation. "
        "Put the note on the fragment where the construction appears. At most one note per item.\n"
        "6. If ref is empty, translate src yourself.\n"
        "7. The id in your reply must be the same integer you received.\n"
        f"{example_line}"
        "Reply with JSON only, no markdown: "
        "{\"items\":[{\"id\":0,\"fragments\":[{\"src\":\"\",\"tr\":\"\",\"words\":[{\"w\":\"\",\"tr\":\"\"}],\"note\":\"\"}]}]}"
    )


class ModelUnavailable(RuntimeError):
    pass


class NetworkUnavailable(RuntimeError):
    pass


class ModelPool:
    def __init__(self, models, retries=8, base_url=PROVIDERS["groq"]["base"]):
        self.models = list(models)
        self.retries = retries
        self.base_url = base_url
        self.idx = 0
        self.lock = threading.Lock()

    def current(self):
        with self.lock:
            return self.models[self.idx]

    def advance(self, failed):
        with self.lock:
            if self.models[self.idx] == failed and self.idx < len(self.models) - 1:
                self.idx += 1
                print(f"  switching model: {failed} -> {self.models[self.idx]}")
            return self.models[self.idx]

    def tag(self):
        return self.models[0]


def list_models(base_url, key):
    try:
        r = requests.get(base_url.rstrip("/") + "/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
    except (requests.RequestException, ValueError) as e:
        print(f"  could not list models: {type(e).__name__}: {str(e)[:150]}")
        return []
    ids = []
    for m in data:
        mid = m.get("id") if isinstance(m, dict) else None
        if not mid:
            continue
        mid = mid[len("models/"):] if mid.startswith("models/") else mid
        if not any(w in mid.lower() for w in SKIP_MODEL_WORDS):
            ids.append(mid)
    return sorted(set(ids))


GROQ_PREFERRED = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]
OLLAMA_PREFERRED = ["qwen2.5", "qwen3", "llama3", "gemma", "mistral", "phi"]


def build_pool(args, key):
    base = args.base
    if args.model:
        return ModelPool([args.model], base_url=base)

    ids = list_models(base, key)

    if args.provider == "groq":
        models = [m for m in GROQ_PREFERRED if m in ids]
        if not models and ids:
            models = ids[:3]
        if not models:
            models = list(GROQ_PREFERRED)
    else:
        if not ids:
            sys.exit("Ollama has no models, or is not running. Install one first, for example: ollama pull qwen2.5:7b")
        models = []
        for pref in OLLAMA_PREFERRED:
            models += [i for i in ids if i.lower().startswith(pref) and i not in models]
        models += [i for i in ids if i not in models]
        models = models[:3]

    print("Models (tried in this order, switches automatically if one fails): " + ", ".join(models))
    return ModelPool(models, base_url=base)


def call_llm(key, model, system, items, retries=5, base_url=PROVIDERS["groq"]["base"]):
    body = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 8000,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Connection": "close",
    }
    url = base_url.rstrip("/") + "/chat/completions"
    net_fail = 0
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=240)
        except requests.RequestException as e:
            net_fail += 1
            print(f"  warning: {model}: network error {type(e).__name__}: {str(e)[:120]} (attempt {attempt + 1}/{retries})")
            time.sleep(min(2 ** attempt, 30))
            continue
        if r.status_code == 402:
            raise RuntimeError(
                "The service says this account has no credits or quota left (HTTP 402). "
                "Use --provider ollama instead, which runs on your computer and costs nothing."
            )
        if r.status_code in (401, 403):
            raise RuntimeError(f"The service refused the request ({r.status_code}): {r.text[:300]}")
        if r.status_code == 404:
            raise ModelUnavailable(f"Model not found: {model}")
        if r.status_code == 400 and "max_tokens" in r.text.lower() and body["max_tokens"] > 4096:
            body["max_tokens"] = 4096
            continue
        if r.status_code == 429 or r.status_code >= 500:
            print(f"  warning: {model}: HTTP {r.status_code}, retrying")
            try:
                wait = min(float(r.headers.get("Retry-After", 0)), 60)
            except ValueError:
                wait = 0
            time.sleep(max(wait, 2 ** attempt * 2))
            continue
        try:
            return r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError):
            print(f"  warning: {model}: unexpected response {r.status_code}: {r.text[:150]!r}")
            time.sleep(2 ** attempt * 2)
    if net_fail == retries:
        raise NetworkUnavailable(base_url)
    return None


def _to_id(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


_ITEM_START = re.compile(r'\{\s*"id"\s*:')


def parse_reply(text):
    if not text:
        return {}
    text = re.sub(r"```(?:json)?", "", text)
    out = {}

    a, b = text.find("{"), text.rfind("}")
    if 0 <= a < b:
        try:
            data = json.loads(text[a:b + 1])
            items = data.get("items", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for it in items:
                if isinstance(it, dict) and "id" in it:
                    k = _to_id(it["id"])
                    if k is not None:
                        out[k] = it.get("fragments")
            if out:
                return out
        except ValueError:
            pass

    dec = json.JSONDecoder()
    for m in _ITEM_START.finditer(text):
        try:
            it, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(it, dict):
            k = _to_id(it.get("id"))
            if k is not None:
                out[k] = it.get("fragments")
    return out


def _core(s):
    s = unicodedata.normalize("NFKC", s).casefold()
    return "".join(c for c in s if c.isalnum())


def _slice_by_core(src, frag_srcs):
    contrib = []
    for idx, ch in enumerate(src):
        c = _core(ch)
        if c:
            contrib.append((idx, c))
    targets = [_core(f) for f in frag_srcs]
    if "".join(c for _, c in contrib) != "".join(targets):
        return None
    out, k, start = [], 0, 0
    for n, tg in enumerate(targets):
        if n == len(targets) - 1:
            out.append(src[start:].strip())
            break
        need, got = len(tg), 0
        if need == 0:
            return None
        while got < need and k < len(contrib):
            got += len(contrib[k][1])
            k += 1
        if got != need:
            return None
        end = contrib[k - 1][0] + 1
        while end < len(src) and not src[end].isalnum() and not src[end].isspace():
            end += 1
        out.append(src[start:end].strip())
        start = end
    if any(not s for s in out):
        return None
    return out


def clean_fragments(frags, src):
    if not isinstance(frags, list) or not frags:
        return None
    cleaned = []
    for f in frags:
        if not isinstance(f, dict) or not isinstance(f.get("src"), str) or not f["src"].strip():
            continue
        words = []
        for w in f.get("words") or []:
            if isinstance(w, dict) and isinstance(w.get("w"), str) and isinstance(w.get("tr"), str):
                if w["w"].strip() and w["tr"].strip():
                    words.append((w["w"].strip(), w["tr"].strip()))
        note = f.get("note")
        cleaned.append({
            "src": f["src"].strip(),
            "tr": f["tr"].strip() if isinstance(f.get("tr"), str) else "",
            "words": words,
            "note": note.strip() if isinstance(note, str) else "",
        })
    if not cleaned:
        return None

    sliced = _slice_by_core(src, [f["src"] for f in cleaned])
    if sliced:
        for f, s in zip(cleaned, sliced):
            f["src"] = s
        return cleaned

    tr = " ".join(f["tr"] for f in cleaned if f["tr"])
    if not tr:
        return None
    words = [w for f in cleaned for w in f["words"]][:6]
    note = next((f["note"] for f in cleaned if f["note"]), "")
    return [{"src": " ".join(src.split()), "tr": tr, "words": words, "note": note}]


def unit_key(model, level, o, t):
    return hashlib.sha1(f"{PROMPT_VERSION}|{model}|{level}|{o}|{t}".encode("utf-8")).hexdigest()


def annotate_units(units, key, pool, level, orig_name, trans_name, cache_path, batch, workers, orig_lang=None, trans_lang=None):
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as fh:
                cache = json.load(fh)
        except ValueError:
            cache = {}

    lock = threading.Lock()
    system = system_prompt(level, orig_name, trans_name, orig_lang, trans_lang)
    tag = f"{pool.tag()}|{orig_name}>{trans_name}"
    rounds = max(3, min(len(pool.models), 6))
    pending = [i for i, (o, t) in enumerate(units) if unit_key(tag, level, o, t) not in cache]
    batches = [pending[k:k + batch] for k in range(0, len(pending), batch)]
    print(f"  {len(units)} units, {len(units) - len(pending)} cached, {len(pending)} to send in {len(batches)} requests")
    done = [0]

    def save():
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(cache, fh, ensure_ascii=False)

    def attempt(idxs):
        model = pool.current()
        items = [{"id": i, "src": units[i][0], "ref": units[i][1]} for i in idxs]
        try:
            reply = call_llm(key, model, system, items, retries=pool.retries, base_url=pool.base_url)
        except ModelUnavailable:
            reply = None
        result = parse_reply(reply)
        if reply is None:
            print(f"  warning: {model}: no reply from the API")
        elif not result:
            print(f"  warning: {model}: reply was not valid JSON: {reply[:150]!r}")
        left = []
        for i in idxs:
            frags = clean_fragments(result.get(i), units[i][0])
            if frags is None:
                left.append(i)
            else:
                with lock:
                    cache[unit_key(tag, level, *units[i])] = frags
        if left and result:
            print(f"  warning: {model}: {len(left)} of {len(idxs)} units failed validation")
        return model, result, left

    def work(idxs):
        todo = list(idxs)
        started = time.time()
        for _ in range(rounds):
            if not todo:
                break
            print(f"  sending {len(todo)} units to {pool.current()}...", flush=True)
            started = time.time()
            model, result, left = attempt(todo)
            if len(left) == len(todo):
                pool.advance(model)
            todo = left

        if todo and len(idxs) > 1:
            print(f"  retrying {len(todo)} stubborn units one by one...", flush=True)
            still = []
            for i in todo:
                _, _, left = attempt([i])
                still += left
            todo = still

        with lock:
            done[0] += 1
            save()
            print(f"  request {done[0]}/{len(batches)} done in {time.time() - started:.0f}s ({len(todo)} units fell back to plain translation)", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(work, b) for b in batches]
        try:
            for f in as_completed(futures):
                f.result()
        except KeyboardInterrupt:
            ex.shutdown(wait=False, cancel_futures=True)
            with lock:
                save()
            print("\nStopped. Progress is saved; run the same command again to continue.", flush=True)
            os._exit(130)
        except NetworkUnavailable as e:
            ex.shutdown(wait=False, cancel_futures=True)
            with lock:
                save()
            sys.exit(
                f"\nStopped: cannot reach {e}. This is a problem with your computer's internet connection "
                "(proxy, VPN or firewall), not with the model. Progress is saved; run the same command "
                "again once the connection works and it will continue where it stopped."
            )
        except RuntimeError as e:
            ex.shutdown(wait=False, cancel_futures=True)
            with lock:
                save()
            sys.exit(f"\nStopped: {e}")

    return [cache.get(unit_key(tag, level, o, t)) for o, t in units]


def strip_stray_quotes(t):
    for op, cl in _QUOTE_PAIRS:
        if t.count(op) != t.count(cl):
            t = t.replace(op, "").replace(cl, "")
    if t.count('"') % 2:
        t = t.replace('"', "")
    return t.strip()


def _piece(s):
    s = strip_stray_quotes(" ".join(s.split()))
    return s.rstrip(" ,;:.!?…–—-")


def span(text):
    return f'<span class="tr" style="color:{GREEN}">({text})</span>'


def render_plain(o, t):
    if not t:
        return html.escape(o)
    m = _TRAIL.search(o)
    body, tail = (o[:m.start()], m.group(1)) if m else (o, "")
    t = _piece(t)
    return f"{html.escape(body.rstrip())} {span(html.escape(t))}{html.escape(tail)}"


def render_fragment(fr):
    src = fr["src"]
    m = _TAIL.search(src)
    body, tail = (src[:m.start()], m.group(1).strip()) if m else (src, "")
    parts = []
    tr = _piece(fr["tr"]) if fr["tr"] else ""
    if tr:
        parts.append(html.escape(tr))
    for w, wtr in fr["words"]:
        w, wtr = _piece(w), _piece(wtr)
        if w and wtr:
            parts.append(f"<i>{html.escape(w)}</i> — {html.escape(wtr)}")
    note = _piece(fr["note"]) if fr["note"] else ""
    if note:
        parts.append(html.escape(note))
    if not parts or not body.strip():
        return html.escape(src)
    return f"{html.escape(body.rstrip())} {span('; '.join(parts))}{html.escape(tail)}"


def render_unit(o, t, frags):
    if frags:
        return " ".join(render_fragment(f) for f in frags)
    return render_plain(o, t)


def build_epub(units, rendered, output_path, orig_lang, title, group=3, ipa=None):
    book = epub.EpubBook()
    book.set_identifier(f"frank_{orig_lang}_{int(hashlib.sha1(title.encode()).hexdigest(), 16) % 10**8}")
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
            f"span.tr{{color:{GREEN};}}"
            "p.ipa{color:#000;}"
        ),
    )
    book.add_item(css)

    per_file = 60
    chapters = []
    for start in range(0, len(units), per_file):
        body = []
        for g in range(start, min(start + per_file, len(units)), group):
            idx = range(g, min(g + group, start + per_file, len(units)))
            mixed = " ".join(rendered[i] for i in idx)
            clean_text = " ".join(units[i][0] for i in idx)
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


def align_books(orig_epub, trans_epub, debug_tsv=None, orig_lang=None, trans_lang=None):
    print("Step 1: extracting text...")
    orig_sents = blocks_to_sentences(extract_blocks_from_epub(orig_epub))
    trans_sents = blocks_to_sentences(extract_blocks_from_epub(trans_epub))
    print(f"  original: {len(orig_sents)} sentences, translation: {len(trans_sents)} sentences")
    if not orig_sents or not trans_sents:
        sys.exit("Error: one of the books produced no text.")
    if orig_lang:
        check_language(orig_sents, orig_lang, "original")
    if trans_lang:
        check_language(trans_sents, trans_lang, "translation")

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

    units = []
    for i0, i1, j0, j1 in spans:
        o = " ".join(orig_sents[i0:i1])
        t = " ".join(trans_sents[j0:j1])
        if o:
            units.append((o, t))

    no_ref = sum(1 for o, t in units if not t)
    print(f"  {len(units)} units, {no_ref} without a translation from the second EPUB "
          f"(the model will translate those itself)")
    return units


def create_book(orig_epub, trans_epub, output_epub, orig_lang, trans_lang, level, key, args):
    print(f"\n[{orig_lang.upper()}] {orig_epub} -> [{trans_lang.upper()}] {trans_epub}, level {level}")
    units = align_books(orig_epub, trans_epub, args.debug_align, orig_lang, trans_lang)
    if args.limit:
        units = units[:args.limit]
        print(f"  test mode: only the first {len(units)} units")

    if not args.limit and len(units) > 100:
        requests_needed = -(-len(units) // args.batch)
        try:
            a = input(
                f"The book has {len(units)} units ({requests_needed} requests). "
                "Type 'all' to process everything, or a number to process only the first N units [30]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            a = "all"
        if a != "all":
            args.limit = int(a) if a.isdigit() and int(a) > 0 else 30
            units = units[:args.limit]
            print(f"  test mode: only the first {len(units)} units")

    print("Step 4: asking the model for glosses and comments...")
    annotated = annotate_units(
        units, key, args.pool, level,
        LANG_NAMES.get(orig_lang, orig_lang), LANG_NAMES.get(trans_lang, trans_lang),
        args.cache or output_epub + ".cache.json", args.batch, args.workers,
        orig_lang, trans_lang,
    )

    missing = sum(1 for (o, t), fr in zip(units, annotated) if not fr and not t)
    plain = sum(1 for fr in annotated if not fr)
    print(f"  {plain} of {len(units)} units have no annotations; "
          f"{missing} of those have no translation at all (run the same command again to retry them)")

    rendered = [render_unit(o, t, fr) for (o, t), fr in zip(units, annotated)]

    print("Step 5: writing EPUB...")
    ipa = make_ipa_hinter() if (not args.no_ipa and orig_lang == "en") else None
    build_epub(units, rendered, output_epub, orig_lang,
               f"Bilingual Book ({orig_lang.upper()} - {trans_lang.upper()}, {level})",
               group=args.group, ipa=ipa)
    print(f"Done: {output_epub}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orig", nargs="?")
    ap.add_argument("trans", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--orig-lang", choices=list(LANG_NAMES), help="language of the book you are learning; asked if omitted")
    ap.add_argument("--trans-lang", choices=list(LANG_NAMES), help="language of the translation and comments; asked if omitted")
    ap.add_argument("--provider", choices=list(PROVIDERS), help="service to use; asked interactively if omitted")
    ap.add_argument("--base-url", help="custom OpenAI-compatible endpoint, overrides the provider's address")
    ap.add_argument("--level", choices=list(LEVELS), help="reader level; asked interactively if omitted")
    ap.add_argument("--model", help="force one specific model id; chosen automatically if omitted")
    ap.add_argument("--batch", type=int, help="sentences per request (default 4, or 2 for Ollama)")
    ap.add_argument("--workers", type=int, help="parallel requests (default 1)")
    ap.add_argument("--limit", type=int, help="process only the first N units (for testing)")
    ap.add_argument("--group", type=int, default=3, help="sentences per block")
    ap.add_argument("--cache", help="cache file (default: <output>.cache.json)")
    ap.add_argument("--no-ipa", action="store_true", help="disable the pronunciation line (English only)")
    ap.add_argument("--debug-align", metavar="FILE.tsv", help="write the sentence alignment to a TSV")
    args = ap.parse_args()

    orig_lang = args.orig_lang or choose_language("Language of the book you are learning (the original text):")
    trans_lang = args.trans_lang or choose_language("Language of your translation and comments:", exclude=orig_lang)
    if orig_lang == trans_lang:
        sys.exit("The two languages must be different.")
    orig_path = args.orig or find_book(orig_lang, "original")
    trans_path = args.trans or find_book(trans_lang, "translation")

    level = args.level or choose_level()
    args.provider = args.provider or choose_provider()
    prov = PROVIDERS[args.provider]
    args.base = (args.base_url or prov["base"]).rstrip("/")
    key = get_api_key(prov)
    args.pool = build_pool(args, key)
    if args.batch is None:
        args.batch = 2 if args.provider == "ollama" else 4
    if args.workers is None:
        args.workers = 1

    out = args.out or f"{orig_lang}_{trans_lang}_{level}_frank.epub"
    create_book(orig_path, trans_path, out, orig_lang, trans_lang, level, key, args)


if __name__ == "__main__":
    main()

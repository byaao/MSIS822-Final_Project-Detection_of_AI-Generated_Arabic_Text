# arabic_preprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import regex as re

# NLTK stopwords (optional; we fall back gracefully if the corpus isn't present)
try:
    import nltk
    from nltk.corpus import stopwords as _nltk_stopwords

    try:
        _ = _nltk_stopwords.words("arabic")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

# CAMeL Tools (required for lemmatization; the rest of the pipeline works without lemmas)
try:
    from camel_tools.utils.normalize import (
        normalize_unicode,          # Unicode normalization
        normalize_alef_ar,          # أ/إ/آ/ٱ → ا
        normalize_alef_maksura_ar,  # ى → ي
    )
    # Remove diacritics (tashkeel)
    from camel_tools.utils.dediac import dediac_ar
    from camel_tools.tokenizers.word import simple_word_tokenize
    # Lemmatizer (MLE disambiguation)
    from camel_tools.disambig.mle import MLEDisambiguator
    _HAS_CAMEL = True
except Exception:
    _HAS_CAMEL = False

# ---------------------------------------------------------------------
# Proclitic constants (light heuristic, no full morphological segmentation)
# ---------------------------------------------------------------------
# و ف ب ك ل ل (note doubled ل to catch definite article combos)
PROCLITIC_CHARS = "وفبكلل"
DEF_ARTICLE = "ال"

# ---------------------------------------------------------------------
# Normalization options
# ---------------------------------------------------------------------


@dataclass
class NormalizeOptions:
    # Core text normalization toggles
    unicode_nfkc: bool = True
    strip_diacritics: bool = True
    unify_alef_variants: bool = True           # أ/إ/آ/ٱ → ا
    tame_hamza_carriers: bool = False          # map ؤ/ئ → ء (standalone hamza)
    map_alef_maksura_to_ya: bool = True        # ى → ي
    remove_tatweel: bool = True                # remove ـ
    normalize_punct: bool = True               # Arabic → ASCII punctuation
    normalize_digits: bool = True              # Arabic-Indic/Persian → ASCII digits
    collapse_whitespace: bool = True

    # Tokenization / segmentation
    # naive proclitic stripping (و، ف، ب، ك، ل …)
    strip_proclitics: bool = False
    # keep punctuation tokens from tokenizer
    keep_punct_tokens: bool = True


# ---------------------------------------------------------------------
# Stopwords (normalized to match pipeline)
# ---------------------------------------------------------------------
# A tiny fallback set if NLTK isn't available; you can replace/extend as needed.
_FALLBACK_AR_STOPWORDS = {
    "في", "من", "على", "عن", "إلى", "الى", "و", "ف", "ب", "ك", "ل", "لا", "ما", "ماذا",
    "لم", "لن", "لكي", "أو", "او", "ثم", "إن", "ان", "أن", "إنما", "لكن", "بل", "حتى", "هناك",
    "هنا", "هذا", "هذه", "ذلك", "تلك", "أولئك", "اولئك", "كان", "كانت", "يكون", "يكونون",
}


def _load_raw_stopwords() -> Iterable[str]:
    if _HAS_NLTK:
        try:
            return set(_nltk_stopwords.words("arabic"))
        except Exception:
            return _FALLBACK_AR_STOPWORDS
    return _FALLBACK_AR_STOPWORDS


# ---------------------------------------------------------------------
# Character maps for optional normalization
# ---------------------------------------------------------------------
_PUNCT_MAP = str.maketrans({
    "،": ",",
    "؛": ";",
    "؟": "?",
    "«": '"', "»": '"',
    "“": '"', "”": '"', "„": '"',
    "‘": "'", "’": "'",
    "‹": "'", "›": "'",
})

_DIGIT_MAP = str.maketrans({
    # Arabic-Indic
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    # Eastern Arabic-Indic (Persian)
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})

# ---------------------------------------------------------------------
# Lazy CAMeL MLE loader
# ---------------------------------------------------------------------
_MLE = None


def _mle():
    if not _HAS_CAMEL:
        raise ImportError(
            "camel_tools is required for lemmatization. "
            "Install with: pip install camel-tools"
        )
    global _MLE
    if _MLE is None:
        _MLE = MLEDisambiguator.pretrained()  # Default MSA model
    return _MLE


# ---------------------------------------------------------------------
# Core normalization helpers
# ---------------------------------------------------------------------
_TATWEEL_RE = re.compile(r"\u0640+")  # ـ
_WS_RE = re.compile(r"\s+")


def _apply_char_maps(text: str, opts: NormalizeOptions) -> str:
    if opts.normalize_punct:
        text = text.translate(_PUNCT_MAP)
    if opts.normalize_digits:
        text = text.translate(_DIGIT_MAP)
    if opts.remove_tatweel:
        text = _TATWEEL_RE.sub("", text)
    return text


def _tame_hamza_carriers(text: str) -> str:
    # Map hamza-on-waaw/ya
    # This is a lossy but sometimes helpful step for matching.
    return text.replace("ؤ", "و").replace("ئ", "ى")


def _normalize_unicode_nfkc(text: str) -> str:
    if not _HAS_CAMEL:
        # Fallback to Python's unicodedata if CAMeL isn't present
        import unicodedata
        return unicodedata.normalize("NFKC", text)
    return normalize_unicode(text, compatibility=True)


def _norm_common(text: str, opts: NormalizeOptions) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # Unicode normalization
    if opts.unicode_nfkc:
        text = _normalize_unicode_nfkc(text)

    # Optional char maps
    text = _apply_char_maps(text, opts)

    # Diacritics
    if opts.strip_diacritics and _HAS_CAMEL:
        text = dediac_ar(text)

    # Alef variants
    if opts.unify_alef_variants and _HAS_CAMEL:
        text = normalize_alef_ar(text)

    # Hamza carriers
    if opts.tame_hamza_carriers:
        text = _tame_hamza_carriers(text)

    # Alef Maqṣūra → Ya
    if opts.map_alef_maksura_to_ya and _HAS_CAMEL:
        text = normalize_alef_maksura_ar(text)

    # Whitespace
    if opts.collapse_whitespace:
        text = _WS_RE.sub(" ", text).strip()

    return text

# ---------------------------------------------------------------------
# Stopwords prepared with the same normalization rules
# ---------------------------------------------------------------------


def _build_stopwords(opts: NormalizeOptions) -> set[str]:
    raw = _load_raw_stopwords()
    return {_norm_common(w, opts) for w in raw if w}

# ---------------------------------------------------------------------
# Tokenization & optional proclitic stripping
# ---------------------------------------------------------------------


def tokenize(text: str, opts: Optional[NormalizeOptions] = None) -> List[str]:
    """
    Tokenize Arabic text using CAMeL simple tokenizer if available;
    otherwise a light regex fallback. Honors opts.keep_punct_tokens.
    """
    opts = opts or NormalizeOptions()
    if not text:
        return []

    if _HAS_CAMEL:
        toks = [t for t in simple_word_tokenize(text) if t.strip()]
        if not opts.keep_punct_tokens:
            toks = [t for t in toks if re.search(r"\p{L}|\p{N}", t)]
        return toks

    # Fallback regex-based tokenization
    pattern = r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]"
    toks = re.findall(pattern, text)
    if not opts.keep_punct_tokens:
        toks = [t for t in toks if re.search(r"\p{L}|\p{N}", t)]
    return toks


def _strip_proclitics_token(token: str) -> List[str]:
    """
    Very light heuristic: split one or more proclitics from the beginning
    and separate the Arabic definite article if present.
    Examples:
        "والكتاب" -> ["و", "ال", "كتاب"]
        "فبالمدرسة" -> ["ف", "ب", "ال", "مدرسة"]
    """
    if not token or not re.match(r"^\p{Arabic}", token):
        return [token]

    clitics: List[str] = []
    i = 0
    # Consume proclitic chars greedily
    while i < len(token) and token[i] in PROCLITIC_CHARS:
        clitics.append(token[i])
        i += 1

    # Definite article
    rest = token[i:]
    if rest.startswith(DEF_ARTICLE):
        clitics.append(DEF_ARTICLE)
        rest = rest[len(DEF_ARTICLE):]

    pieces = clitics + ([rest] if rest else [])
    # Remove empties
    return [p for p in pieces if p]


def maybe_strip_proclitics(tokens: List[str], opts: NormalizeOptions) -> List[str]:
    if not opts.strip_proclitics:
        return tokens
    out: List[str] = []
    for t in tokens:
        out.extend(_strip_proclitics_token(t))
    return out

# ---------------------------------------------------------------------
# Lemmatization (CAMeL MLE)
# ---------------------------------------------------------------------


def lemmas_via_camel(tokens: List[str]) -> List[str]:
    if not tokens:
        return []
    disamb = _mle().disambiguate(tokens)
    lemmas: List[str] = []
    for d in disamb:
        # Defensive: d.analyses may be empty
        if getattr(d, "analyses", None):
            a = d.analyses[0].analysis
            # Prefer 'lex' if available
            lemma = a.get("lex", a.get("lemma", d.word))
        else:
            lemma = d.word
        lemmas.append(lemma)
    return lemmas

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def normalize_text(text: str, opts: Optional[NormalizeOptions] = None) -> str:
    """Normalize Arabic text according to the provided NormalizeOptions."""
    return _norm_common(text, opts or NormalizeOptions())


def preprocess_arabic(
    text: str,
    *,
    remove_stopwords: bool = True,
    to_lemmas: bool = True,
    opts: Optional[NormalizeOptions] = None,
) -> str:
    """
    Complete Arabic preprocessing pipeline.

    Steps (depending on options):
      - Unicode NFKC, punctuation and digit normalization
      - Remove tatweel, diacritics (tashkeel), and unify Alef variants
      - Tame hamza carriers (map و/ى → ؤ/ئ)
      - Map Alef Maqṣūra (ى) → Ya (ي)
      - Tokenize (CAMeL if available; regex fallback)
      - Optional light proclitic splitting
      - Optional lemmatization via CAMeL MLE
      - Optional stopword removal (stoplist normalized with same opts)
    """
    opts = opts or NormalizeOptions()

    # Normalize
    s = normalize_text(text, opts=opts)

    # Tokenize (+ optional proclitic split)
    tokens = tokenize(s, opts=opts)
    tokens = maybe_strip_proclitics(tokens, opts)

    # Lemmas (if requested and CAMeL installed)
    if to_lemmas:
        if not _HAS_CAMEL:
            raise ImportError(
                "camel_tools is required for lemmatization. "
                "Install with: pip install camel-tools"
            )
        tokens = lemmas_via_camel(tokens)
        # Normalize lemmas with SAME character rules to keep things consistent
        tokens = [normalize_text(t, opts=opts) for t in tokens]

    # Stopwords
    if remove_stopwords:
        stopset = _build_stopwords(opts)
        tokens = [t for t in tokens if t not in stopset]

    return " ".join(tokens)

# ---------------------------------------------------------------------
# Convenience: one-shot call with custom switches
# ---------------------------------------------------------------------


def make_options(
    *,
    unicode_nfkc: bool = True,
    strip_diacritics: bool = True,
    unify_alef_variants: bool = True,
    tame_hamza_carriers: bool = False,
    map_alef_maksura_to_ya: bool = True,
    remove_tatweel: bool = True,
    normalize_punct: bool = True,
    normalize_digits: bool = True,
    collapse_whitespace: bool = True,
    strip_proclitics: bool = False,
    keep_punct_tokens: bool = True,
) -> NormalizeOptions:
    """Helper to build NormalizeOptions with keyword switches."""
    return NormalizeOptions(
        unicode_nfkc=unicode_nfkc,
        strip_diacritics=strip_diacritics,
        unify_alef_variants=unify_alef_variants,
        tame_hamza_carriers=tame_hamza_carriers,
        map_alef_maksura_to_ya=map_alef_maksura_to_ya,
        remove_tatweel=remove_tatweel,
        normalize_punct=normalize_punct,
        normalize_digits=normalize_digits,
        collapse_whitespace=collapse_whitespace,
        strip_proclitics=strip_proclitics,
        keep_punct_tokens=keep_punct_tokens,
    )


__all__ = [
    "NormalizeOptions",
    "make_options",
    "normalize_text",
    "tokenize",
    "lemmas_via_camel",
    "preprocess_arabic",
    "PROCLITIC_CHARS",
    "DEF_ARTICLE",
]

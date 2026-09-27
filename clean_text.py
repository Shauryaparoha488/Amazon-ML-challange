"""
Phase 2: Text cleaning / normalization for the Amazon ML entity resolution challenge.

Built directly from the Phase 1 EDA noise catalog:
- Case/punctuation noise, junk prefixes (>>, <<, #, ...)
- Legal suffixes that move or vanish (Inc/LLC/Ltd/Pvt/Corp/PLLC + French SARL/SCI/SAS)
- Honorifics (Sri, Smt, Mr, Shri)
- Website-ization (.com, www.)
- Diacritics / lookalike chars (o/0, l/1, etc. — light touch, NFKC handles most)
- Street-type + admin abbreviations (US, India, France)
- PO Box / unit noise
- Landmark phrases (India) - dropped after street tokens captured

No external lookups, no internet transliteration — everything is a static in-code table.

Produces, per record:
- clean_name        : normalized name, suffixes/honorifics/junk stripped
- name_tokens       : sorted core tokens (for blocking keys / Jaccard)
- clean_address     : normalized address, abbreviations expanded, landmarks dropped
- house_number      : extracted leading house number if present (else None)
- addr_tokens       : sorted address tokens (for blocking keys / Jaccard)

Column cleaning in process_dataframe() is pandas vectorized (.str.lower / .str.replace).
Scalar clean_name / clean_address wrap a 1-row Series so smoke tests share the same rules.
"""

from __future__ import annotations

import re

import pandas as pd

# ---------------------------------------------------------------------------
# Lookup tables (static, in-code — no external calls)
# ---------------------------------------------------------------------------

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt", "private",
    "limited", "ltd", "llc", "l.l.c", "pllc", "p.l.l.c",
    "inc", "incorporated",
    "corporation", "corp", "co", "company", "pc", "p.c",
    "dmd", "llp", "l.l.p",
    # French forms
    "sarl", "sci", "sas", "s.a.s", "s.a.r.l",
]

HONORIFICS = ["sri", "smt", "shri", "mr", "mrs", "ms", "dr"]

# generic filler words — dropped ONLY after suffix strip, kept as a short list
# per EDA note: be careful, don't over-strip (Associates vs Madison example)
GENERIC_FILLERS = ["group", "center", "centre"]
_GENERIC_FILLER_SET = set(GENERIC_FILLERS)

STREET_ABBR = {
    "st": "street", "st.": "street",
    "rd": "road", "rd.": "road",
    "dr": "drive", "dr.": "drive",
    "ave": "avenue", "ave.": "avenue",
    "ln": "lane", "ln.": "lane",
    "blvd": "boulevard", "blvd.": "boulevard",
    "r": "rue", "r.": "rue",  # French
}

US_STATE_ABBR = {
    "oh": "ohio", "nc": "north carolina",
    # extend with full 50-state table as needed
}

INDIA_STATE_ABBR = {
    "mh": "maharashtra", "ka": "karnataka", "dl": "delhi",
    "महाराष्ट्र": "maharashtra", "दिल्ली": "delhi",
}

CITY_ALIASES = {
    "bangalore": "bengaluru",
    "poona": "pune",
}

FRANCE_TOKENS = {"bis", "boulevard", "rue"}  # kept as-is, just recognized

# After address tokens have trailing/leading dots stripped, dotted dict keys
# are redundant; keep them so scalar-style lookups stay equivalent.
_ADDR_EXPAND = {**STREET_ABBR, **US_STATE_ABBR, **INDIA_STATE_ABBR, **CITY_ALIASES}
_ADDR_EXPAND_KEYS = sorted(_ADDR_EXPAND, key=len, reverse=True)

_LEGAL_SORTED = sorted(LEGAL_SUFFIXES, key=len, reverse=True)
_LEGAL_ALT = r"\b(?:" + "|".join(re.escape(s) for s in _LEGAL_SORTED) + r")\b"
_HONORIFIC_ALT = r"\b(?:" + "|".join(re.escape(s) for s in HONORIFICS) + r")\b"
_ADDR_REGEX_MAP = {
    rf"\b{re.escape(src)}\b": dst
    for src, dst in sorted(_ADDR_EXPAND.items(), key=lambda kv: len(kv[0]), reverse=True)
}

JUNK_CHARS_RE = r"(>>|<<|#|\.\.\.|--|<null>)"
WEBSITE_WWW_RE = r"\bwww\."
WEBSITE_COM_RE = r"\b([\w\-]+)\.com\b"
MULTISPACE_RE = r"\s+"
DUPLICATE_WORD_RE = r"\b(\w+)( \1\b)+"
LANDMARK_RE = r"\b(near|opp\.?|opposite)\b[^,]*"
PO_BOX_RE = r"\b(p\.?\s*o\.?\s*box|pmb|unit)\s*#?\s*\d*\b"
DOUBLE_COMMA_RE = r",\s*,"
PUNCT_NAME_RE = r"[^\w\s.\-]"
PUNCT_ADDR_RE = r"[^\w\s.\-,]"
LEADING_TOKEN_DOTS_RE = r"(?:(?<=\s)|^)\.+"
TRAILING_TOKEN_DOTS_RE = r"\.+(?=\s|$)"
HOUSE_NUMBER_RE = r"^(\d+)\b"

# Combining marks left after NFKC (Latin + common blocks). A full
# unicodedata.combining character class is thousands of code points and
# makes Series.str.replace unusably slow on millions of rows.
_COMBINING_RE = re.compile(
    r"[\u0300-\u036f\u0483-\u0489\u0591-\u05bd\u0610-\u061a"
    r"\u064b-\u065f\u0670\u06d6-\u06ed"
    r"\u1ab0-\u1afe\u1dc0-\u1dff\u20d0-\u20f0\ufe20-\ufe2f]"
)


def _series_base_clean(s: pd.Series, keep_commas: bool = False) -> pd.Series:
    """Vectorized lowercase + NFKC + junk/punct squeeze. NA -> empty string."""
    s = s.fillna("").astype(str)
    s = s.str.lower()
    s = s.str.normalize("NFKC")
    s = s.str.replace(_COMBINING_RE, "", regex=True)
    s = s.str.replace(JUNK_CHARS_RE, " ", regex=True)
    punct = PUNCT_ADDR_RE if keep_commas else PUNCT_NAME_RE
    s = s.str.replace(punct, " ", regex=True)
    s = s.str.replace(MULTISPACE_RE, " ", regex=True).str.strip()
    return s


def series_clean_name(s: pd.Series) -> pd.Series:
    """Vectorized name normalization (whole column)."""
    s = _series_base_clean(s, keep_commas=False)
    s = s.str.replace(WEBSITE_WWW_RE, "", regex=True)
    s = s.str.replace(WEBSITE_COM_RE, r"\1", regex=True)
    s = s.str.replace(_HONORIFIC_ALT, " ", regex=True)
    s = s.str.replace(_LEGAL_ALT, " ", regex=True)
    s = s.str.replace(MULTISPACE_RE, " ", regex=True).str.strip()
    s = s.str.replace(DUPLICATE_WORD_RE, r"\1", regex=True)
    return s.str.replace(MULTISPACE_RE, " ", regex=True).str.strip()


def series_clean_address(s: pd.Series) -> pd.Series:
    """Vectorized address normalization (whole column)."""
    s = _series_base_clean(s, keep_commas=True)
    s = s.str.replace(LANDMARK_RE, "", regex=True)
    s = s.str.replace(DOUBLE_COMMA_RE, ",", regex=True)
    s = s.str.strip(" ,")
    s = s.str.replace(PO_BOX_RE, " ", regex=True)
    s = s.str.replace(",", " ", regex=False)
    s = s.str.replace(MULTISPACE_RE, " ", regex=True).str.strip()
    # Match per-token t.strip('.') before abbreviation lookup
    s = s.str.replace(LEADING_TOKEN_DOTS_RE, "", regex=True)
    s = s.str.replace(TRAILING_TOKEN_DOTS_RE, "", regex=True)
    s = s.replace(_ADDR_REGEX_MAP, regex=True)
    return s.str.replace(MULTISPACE_RE, " ", regex=True).str.strip()


def _one(raw) -> pd.Series:
    return pd.Series(["" if pd.isna(raw) else raw])


def clean_name(raw_name: str) -> str:
    """Normalize a business name (scalar wrapper around the vectorized path)."""
    return series_clean_name(_one(raw_name)).iloc[0]


def clean_address(raw_address: str) -> str:
    """Normalize an address (scalar wrapper around the vectorized path)."""
    return series_clean_address(_one(raw_address)).iloc[0]


def name_core_tokens(clean: str) -> frozenset:
    """Core token set for blocking/Jaccard: drop generic fillers, sort/dedupe."""
    tokens = [t for t in clean.split() if t not in _GENERIC_FILLER_SET]
    return frozenset(tokens)


def extract_house_number(clean_addr: str):
    """Extract leading house number if present, else None. Not used as a hard block key
    (EDA shows digit typos: 3250 vs 325), only as a soft feature later."""
    if not clean_addr:
        return None
    m = re.match(HOUSE_NUMBER_RE, clean_addr)
    return m.group(1) if m else None


def addr_core_tokens(clean_addr: str) -> frozenset:
    return frozenset(clean_addr.split())


def _token_strings(series: pd.Series, drop_fillers: bool = False) -> list[str]:
    """Split is vectorized; set join is a cheap Python pass on already-short token lists."""
    lists = series.str.split().tolist()
    out = []
    for toks in lists:
        if not toks:
            out.append("")
            continue
        if drop_fillers:
            toks = [t for t in toks if t not in _GENERIC_FILLER_SET]
        out.append(" ".join(sorted(set(toks))))
    return out


def process_dataframe(
    df: pd.DataFrame,
    name_col="business_name",
    addr_col="business_address",
    copy: bool = True,
) -> pd.DataFrame:
    """Apply full Phase 2 cleaning with vectorized string ops on each column."""
    if copy:
        df = df.copy()
    df["clean_name"] = series_clean_name(df[name_col])
    df["clean_address"] = series_clean_address(df[addr_col])
    df["house_number"] = df["clean_address"].str.extract(HOUSE_NUMBER_RE, expand=False)
    df["name_tokens"] = _token_strings(df["clean_name"], drop_fillers=True)
    df["addr_tokens"] = _token_strings(df["clean_address"], drop_fillers=False)
    return df


if __name__ == "__main__":
    examples = [
        "RAMIREZ EQUITY PARTNERS",
        "Ramirez Equity Partners PC",
        "Pediatric Medicine PLLC",
        "Cornerstone-Trust,-Inc",
        ">> Limited Iconic Msagemegnt",
        "Private Swamiraj Tdaaing Limited",
        "Sri Nadine's Associates-Portland",
        "swamirajtrading.com",
        "Faon Faon Fortis",
    ]
    for ex in examples:
        print(f"{ex!r:45} -> {clean_name(ex)!r}")

    addr_examples = [
        "5559 Orville Avenue, Columbus, OH",
        "Near Fortis Hospital, MG Road, Bangalore, KA",
        "PO BOX 56, Unit 12, Main St",
        "12 Rue de la Paix, Gironde",
    ]
    print()
    for ex in addr_examples:
        print(f"{ex!r:50} -> {clean_address(ex)!r}")

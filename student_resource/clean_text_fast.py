"""
Phase 2: Text cleaning / normalization — fully vectorized version.

Fixes the exact bug Cursor's agent hit and got stuck fixing: the original
per-character Python loop to strip Unicode combining marks
(`"".join(c for c in text if not unicodedata.combining(c))`) is O(total
characters) with a Python function call per character — catastrophically
slow at 5M+ rows. Replaced with a single vectorized regex substitution
after NFKD decomposition, which runs the loop in C instead of Python.

Same cleaning rules as before (built from the Phase 1 EDA noise catalog),
just restructured so every step is a pandas Series.str vectorized op
instead of .apply() row-by-row.
"""

import re
import pandas as pd

# ---------------------------------------------------------------------------
# Lookup tables (static, in-code — no external calls)
# ---------------------------------------------------------------------------

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt", "private",
    "limited", "ltd", "llc", "inc", "incorporated",
    "corporation", "corp", "company", "pc", "dmd",
    "llp", "pllc",
    "sarl", "sci", "sas",  # French
]

HONORIFICS = ["sri", "smt", "shri", "mr", "mrs", "ms", "dr"]
GENERIC_FILLERS = ["group", "center", "centre"]

STREET_ABBR = {
    "st": "street", "rd": "road", "dr": "drive", "ave": "avenue",
    "ln": "lane", "blvd": "boulevard", "r": "rue",
}
US_STATE_ABBR = {"oh": "ohio", "nc": "north carolina"}
INDIA_STATE_ABBR = {"mh": "maharashtra", "ka": "karnataka", "dl": "delhi"}
CITY_ALIASES = {"bangalore": "bengaluru", "poona": "pune"}

ABBR_MAP = {**STREET_ABBR, **US_STATE_ABBR, **INDIA_STATE_ABBR, **CITY_ALIASES}

# ---------------------------------------------------------------------------
# Precompiled patterns (built once, reused across the whole column)
# ---------------------------------------------------------------------------

_COMBINING_MARKS = re.compile(r"[\u0300-\u036f]")
_JUNK_CHARS = re.compile(r"(>>|<<|#|\.\.\.|--|<null>)", re.IGNORECASE)
_STRAY_PUNCT_KEEP_COMMA = re.compile(r"[^\w\s\.\-,]")
_STRAY_PUNCT_NO_COMMA = re.compile(r"[^\w\s\.\-]")
_MULTISPACE = re.compile(r"\s+")
_WEBSITE = re.compile(r"\b(www\.)?([\w\-]+)\.com\b", re.IGNORECASE)

_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_HONORIFIC_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(h) for h in HONORIFICS) + r")\b", re.IGNORECASE
)
_DUPLICATE_WORD = re.compile(r"\b(\w+)( \1\b)+", re.IGNORECASE)

_LANDMARK_PATTERN = re.compile(r"\b(near|opp\.?|opposite)\b[^,]*", re.IGNORECASE)
_PO_BOX_PATTERN = re.compile(r"\b(p\.?\s*o\.?\s*box|pmb|unit)\s*#?\s*\d*\b", re.IGNORECASE)
_HOUSE_NUMBER = re.compile(r"^\s*(\d+)\b")

# single alternation regex covering all abbreviation keys, longest-first
_ABBR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(ABBR_MAP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _abbr_repl(m: re.Match) -> str:
    return ABBR_MAP[m.group(0).lower()]


# ---------------------------------------------------------------------------
# Vectorized cleaning
# ---------------------------------------------------------------------------

def _strip_combining_marks_vectorized(s: pd.Series) -> pd.Series:
    """NFKD decompose (accented char -> base + combining mark), then regex-strip
    the combining marks in one vectorized pass. Replaces the slow per-character loop."""
    decomposed = s.str.normalize("NFKD")
    return decomposed.str.replace(_COMBINING_MARKS, "", regex=True)


def clean_name_series(raw: pd.Series) -> pd.Series:
    s = raw.fillna("").astype(str).str.lower()
    s = _strip_combining_marks_vectorized(s)
    s = s.str.replace(_JUNK_CHARS, " ", regex=True)
    s = s.str.replace(_STRAY_PUNCT_NO_COMMA, " ", regex=True)
    s = s.str.replace(_MULTISPACE, " ", regex=True).str.strip()

    # website-ization: drop .com / www.
    s = s.str.replace(_WEBSITE, lambda m: m.group(2), regex=True)

    # honorifics + legal suffixes (both anywhere in the string, per EDA)
    s = s.str.replace(_HONORIFIC_PATTERN, "", regex=True)
    s = s.str.replace(_SUFFIX_PATTERN, "", regex=True)
    s = s.str.replace(_MULTISPACE, " ", regex=True).str.strip()

    # collapse duplicated consecutive words
    s = s.str.replace(_DUPLICATE_WORD, r"\1", regex=True)
    return s.str.strip()


def clean_address_series(raw: pd.Series) -> pd.Series:
    s = raw.fillna("").astype(str).str.lower()
    s = _strip_combining_marks_vectorized(s)
    s = s.str.replace(_JUNK_CHARS, " ", regex=True)
    s = s.str.replace(_STRAY_PUNCT_KEEP_COMMA, " ", regex=True)
    s = s.str.replace(_MULTISPACE, " ", regex=True).str.strip()

    # landmark phrases bounded to their comma segment
    s = s.str.replace(_LANDMARK_PATTERN, "", regex=True)
    s = s.str.replace(r",\s*,", ",", regex=True)
    s = s.str.strip(", ").str.strip()

    # PO box / unit noise
    s = s.str.replace(_PO_BOX_PATTERN, " ", regex=True)

    # drop commas now, expand abbreviations in one pass
    s = s.str.replace(",", " ", regex=False)
    s = s.str.replace(_ABBR_PATTERN, _abbr_repl, regex=True)

    s = s.str.replace(_MULTISPACE, " ", regex=True).str.strip()
    return s


def extract_house_number_series(clean_addr: pd.Series) -> pd.Series:
    return clean_addr.str.extract(_HOUSE_NUMBER, expand=False)


def name_tokens_series(clean_name: pd.Series) -> pd.Series:
    filler_pattern = re.compile(r"\b(" + "|".join(GENERIC_FILLERS) + r")\b")
    stripped = clean_name.str.replace(filler_pattern, "", regex=True)
    stripped = stripped.str.replace(_MULTISPACE, " ", regex=True).str.strip()
    return stripped.apply(lambda t: frozenset(t.split()) if t else frozenset())


def addr_tokens_series(clean_addr: pd.Series) -> pd.Series:
    return clean_addr.apply(lambda t: frozenset(t.split()) if t else frozenset())


def process_dataframe(df: pd.DataFrame, name_col="business_name", addr_col="business_address") -> pd.DataFrame:
    """Apply full Phase 2 cleaning to a source dataframe — vectorized, safe for millions of rows."""
    df = df.copy()
    df["clean_name"] = clean_name_series(df[name_col])
    df["name_tokens"] = name_tokens_series(df["clean_name"])
    df["clean_address"] = clean_address_series(df[addr_col])
    df["house_number"] = extract_house_number_series(df["clean_address"])
    df["addr_tokens"] = addr_tokens_series(df["clean_address"])
    return df


if __name__ == "__main__":
    examples = pd.Series([
        "RAMIREZ EQUITY PARTNERS",
        "Ramirez Equity Partners PC",
        "Cornerstone-Trust,-Inc",
        ">> Limited Iconic Msagemegnt",
        "Private Swamiraj Tdaaing Limited",
        "Sri Nadine's Associates-Portland",
        "swamirajtrading.com",
        "Faon Faon Fortis",
        "ABC PLLC Dental",
        "Bítcoin Cárissa Enterprises",
    ])
    print("=== NAME CLEANING ===")
    for raw, clean in zip(examples, clean_name_series(examples)):
        print(f"{raw!r:45} -> {clean!r}")

    addr_examples = pd.Series([
        "5559 Orville Avenue, Columbus, OH",
        "Near Fortis Hospital, MG Road, Bangalore, KA",
        "PO BOX 56, Unit 12, Main St",
        "12 Rue de la Paix, Gironde",
    ])
    print("\n=== ADDRESS CLEANING ===")
    for raw, clean in zip(addr_examples, clean_address_series(addr_examples)):
        print(f"{raw!r:50} -> {clean!r}")

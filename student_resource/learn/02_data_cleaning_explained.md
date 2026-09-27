# Phase 2 — What text cleaning is, and why raw text breaks matching

## Why we do this at all

Computers compare **characters**, not “meaning.”

These two rows are the same company in the ground truth:

- Source 1: `Ramirez Equity Partners PC`
- Source 2: `RAMIREZ EQUITY PARTNERS`

A naive check (`left == right`) says they differ because of **case** and the extra **`PC`**. A person would shrug. The model would not.

**Normalization** (also called cleaning) is a set of boring, consistent rewrites so that two spellings of the same business become more alike *before* we score similarity. It does **not** look companies up on the internet (that is forbidden). Every rule is a table or regex inside `clean_text.py`.

The function `process_dataframe()` adds five columns:

| New column | What it is |
|---|---|
| `clean_name` | Normalized name (suffixes/honorifics/junk stripped) |
| `name_tokens` | Unique name words, fillers like `center` dropped — for blocking / Jaccard |
| `clean_address` | Normalized address (abbreviations expanded, landmarks dropped) |
| `house_number` | Leading digits if the cleaned address starts with a number |
| `addr_tokens` | Unique address words — for blocking / Jaccard |

Below, each transformation uses a **real string from this dataset** (EDA samples or ground-truth pairs), then the actual output of `clean_name` / `clean_address`.

---

## Names

### 1. Lowercasing

**Problem:** `RAMIREZ EQUITY PARTNERS` and `Ramirez Equity Partners` are the same words in different cases.

**Before → after**

- `RAMIREZ EQUITY PARTNERS` → `ramirez equity partners`
- `Ramirez Equity Partners PC` → `ramirez equity partners`

Once both sides are lowercase, the remaining difference was only `PC`, which the suffix step removes. Now they match exactly.

---

### 2. Unicode NFKC + stripping combining marks

**Problem:** Copy-paste and other alphabets produce lookalike letters: `Bítcoin` vs `Bitcoin`, `Léarning` vs `Learning`.

**What the code does:** `unicodedata.normalize("NFKC", text)` (a standard “compatibility” form), then drop combining marks (accents that sit on top of a letter).

**Honest result on this data:** `Sandoval Bítcoin` became `sandoval bítcoin` — the `í` stayed, because it is a single precomposed character that NFKC does not always split. Cleaning is not magic; Phase 4 still needs fuzzy similarity (one-character edits). The step still helps when the text *does* use separate accent marks, and it standardizes some compatibility characters.

---

### 3. Junk prefixes and symbols

**Problem:** Scraped names arrive with `>>`, `<<`, `#`, `...`, `--`, `<NULL>`.

**Before → after**

- `Sri >> ss finance private limited` → `ss finance`  
  (`>>` gone; `sri` gone in the honorific step; `private limited` gone in the suffix step)

- `#shivanew` (from a true match to `Shiva New Consultancy Private Limited`) would lose the `#` so later token overlap on `shivanew` is possible.

Without this, the first token of the name is garbage, and blocking keys that use “first word” fail.

---

### 4. Website-ization (`.com` / `www.`)

**Problem:** One source stores the company as a URL-ish blob.

**Before → after**

- `... swamirajtrading.com` → `swamirajtrading`
- `Munozfuturecrest.Com` → `munozfuturecrest`
- `SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com` → `shivshakti vidyalaya overseas shivshakti`

The ground-truth name was `Swamiraj Trading Private Limited`. After cleaning, you still do not get a perfect string match (`swamirajtrading` vs `swamiraj trading`), but you removed `.com` and `www.` so a later “squeeze spaces” or character-level metric can see the overlap. Suffix strip also takes `corporation` off the long Shivshakti example.

---

### 5. Honorifics

**Problem:** Indian records often prepend `Sri`, `Smt`, `Shri`, `Mr`.

**Before → after**

- `Sri >> ss finance private limited` → `ss finance`
- `Mr SWAMIRAJ TRADING PIETODE LIMITED` would drop `mr`, then drop `limited`, leaving the (still typo’d) core `swamiraj trading pietode`.

The honorific is not part of the legal name. Leaving it in makes Jaccard think two extra tokens differ.

---

### 6. Legal suffix stripping (including French forms)

**Problem:** One side says `LLC` / `Inc` / `Private Limited` / `PC` / `DMD`; the other omits it. Sometimes the suffix is **moved to the front**: `Limited Iconic Msagemegnt`.

The table includes US/India forms **and** test-only French ones (`sarl`, `sci`, `sas`) so France is not a special-case filter — just more words to ignore.

**Before → after**

- `Ramirez Equity Partners PC` → `ramirez equity partners`
- `Private Swamiraj Tdaaing Limited` → `swamiraj tdaaing`
- `Pvt. EFS Print Ventures Ltd.` → `. efs print ventures .`  
  (dots that belonged to `Pvt.` / `Ltd.` can remain — a reminder that cleaning is approximate)
- `LLC Moncada Léarning Center` → `moncada léarning center`
- `Saleh, Carissa P., DMD` → `saleh carissa p.`

After this, `Hiya Constructions` / `Hiya Constructions Company` / `Hiya Constructions Corporation` get much closer.

---

### 7. Duplicate consecutive words

**Problem:** `Faon Faon Fortis` and `Heritage Heritage Mntropolian Globa` (true matches to names without the repeat).

**Before → after**

- `Faon Faon Fortis` → `faon fortis`
- `Heritage Heritage Mntropolian Globa` → `heritage mntropolian globa`

---

### 8. Core name tokens (not the same as `clean_name`)

`clean_name` still keeps words like `group` and `center` (the EDA warned: `Western Madison` vs `Western Center` — if you delete `center` *and* `madison` you over-merge).

`name_tokens` drops only a **tiny** filler list (`group`, `center`, `centre`) and stores a **set** of words (order does not matter). That is for **Jaccard** and blocking: `{western, madison}` vs `{western}` still overlap on `western`, without pretending `center` is the company name.

**Example:** `Lyrance Bhav Group` → `clean_name` is `lyrance bhav group`; tokens drop `group`.

Word-order swaps (`Nadine's Associates-Portland` vs `Nadine's Portland Associates`) become the same set after punctuation split (`nadine`, `s`, `portland`, `associates`) — that is why we tokenize instead of comparing one long string.

---

## Addresses

### 9. Lowercase + junk, but **keep commas a little longer**

Landmarks in this dataset sit in their own comma-separated chunk (`Near Fortis Hospital, Bhandup West, Mumbai, ...`). The cleaner keeps commas until it can delete only that chunk, then turns commas into spaces.

---

### 10. Landmark phrases

**Problem:** India addresses add “Near SBI ATM”, “Opp. Ramakrushi Co.” Source 1 may not have that phrase. If you leave it in, token overlap looks worse even when the street is the same.

**Before → after**

- `Near Fortis Hospital, Bhandup West, Mumbai, Maharashtra` → `bhandup west mumbai maharashtra`
- Long Source 1 address with `Near Fortis Hospital` in the middle: the landmark segment is removed; `2505 tower 1 oakwood runwal greens mulund goreagon link road bhandup west mumbai maharashtra` remains.

The regex only eats from `near` / `opp` / `opposite` **until the next comma**, so it should not delete the rest of the city.

---

### 11. PO Box / unit / PMB noise

**Problem:** Source 2 often adds `PO BOX 9271` or `Unit 56` that Source 1 does not have. Units are useful sometimes, but they are inconsistent; the EDA treated them as noise for the *normalized* string.

**Before → after**

- `##7114 PEACEFUL ACRES LN, PO BOX 9271, MECHANICSVILLE CITY, VA` → `7114 peaceful acres lane mechanicsville city va`  
  (`##` junk gone, `PO BOX 9271` gone, `LN` expanded)

---

### 12. Street-type abbreviations

**Problem:** `ST` vs `Street`, `AVE` vs `Avenue`, `RD` vs `Road`, French `R.` vs `Rue`.

**Before → after**

- `910 PAULINE ST, HIGHLANDS, TX` → `910 pauline street highlands tx`
- `Mack Rd, Haltom City, Texas` → `mack road haltom city texas`
- `3844 BATTLEGROUND AVE, GREENSBORO, NC` → `3844 battleground avenue greensboro north carolina`

Now Jaccard on address tokens can count `street` on both sides.

---

### 13. State / city tables (open-set friendly)

**Problem:** `OH` vs `Ohio`, `MH` vs `Maharashtra`, `KA` vs `Karnataka`, `Bangalore` vs `Bengaluru`, `Poona` vs `Pune`. Also Devanagari state names (`महाराष्ट्र`, `दिल्ली`) mapped in-code — **not** an API call.

**Before → after**

- `5559 Orville Avenue, Columbus, OH` → `5559 orville avenue columbus ohio`
- `G.no.32/1, Pune, MH` → `g.no.32 1 pune maharashtra`
- `HN 745 G.NO.32/1, POONA, PUNE, Maharashtra` → `hn 745 g.no.32 1 pune pune maharashtra`
- `# 143, BANGALORE, BENGALURU, Karnataka` → `143 bengaluru bengaluru karnataka`

**Limitation you should know:** the US table in code currently expands only a few codes (e.g. `OH`, `NC`). `TX` in the Pauline Street example stayed `tx`. That is OK: later features can still match `tx` to `tx`. Expanding the table is a Phase 2 improvement, not a change in *idea*.

We still **do not** drop rows whose country is France. French words like `rue` / `bis` are left in place.

---

### 14. Placeholder tokens in the address

**Problem:** `4849 HONEY BEE LANE, <NULL>, VILLAGE OF MAINE, WI`

**After:** `4849 honey bee lane village of maine wi`

`<NULL>` is treated as junk so it does not become a fake token.

---

### 15. House number (soft feature, not a lock)

If the cleaned address starts with digits, we store them in `house_number`.

- `910 pauline street highlands tx` → house `910`
- `Near Fortis...` after landmark strip may have **no** leading number → `None`

EDA showed **typos in house numbers** (`3250` vs `325` Archer; `3844` vs `6844` Battleground). So Phase 3 should **not** require house numbers to be equal. Phase 4 can use “same house number?” as a *feature* (helpful when it agrees, not fatal when it does not).

---

## What cleaning will not fix (so we do not expect miracles)

- **Typos:** `Msagemegnt`, `Tdaaing`, `Onc0logy`, `6loba`, `Sa1eh`. Need edit-distance / fuzzy features.
- **Different trade names:** `Saleh, Carissa P., DMD` vs `Mirairi`. Need address (and still be careful).
- **Indic script vs Latin:** Kannada `ಎಸ್ಎಸ್ ಫೈನಾನ್ಸ್...` does not become `ss finance` without a romanization table we have not fully built. External transliteration APIs are not allowed.
- **Addresses that are truly missing:** Source 3 `nan` for a real match stays empty after cleaning. Compare to the EDA blank rates in `clean_output.txt`. If *new* blanks appear, it means some “addresses” were only landmarks/PO Boxes and the cleaner erased them — worth knowing for blocking.

---

## How this feeds Phase 3 (blocking)

After cleaning, cheap keys become possible, for example:

- same `country` (100% of train true pairs)
- shared tokens from `name_tokens`
- same `house_number` **plus** a city token (soft, recall-oriented)

We still compare those shortlists with proper similarity in Phase 4. Cleaning is the difference between “`ST` never equals `Street`” and “both say `street`.”

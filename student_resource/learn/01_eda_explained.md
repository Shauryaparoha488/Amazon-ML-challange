# Phase 1 — What EDA is, and why we did it first

## The one-sentence version

**EDA** (exploratory data analysis) means *looking at the data before you build a model*, so you do not guess.

This challenge is **entity resolution**: three messy lists of businesses, and we must decide which Source 2 / Source 3 rows are the same company as each Source 1 row. If you skip EDA, you might:

- try to compare every pair of records (impossible at this size)
- hard-code “only US and India” and then fail on France in the test set
- ignore companies that truly have **no** match (those still affect the score)

The script `eda_phase1.py` answered a short list of practical questions. Below, each check is: *what we asked*, *what we found*, *what we do next because of it*.

---

## 1. How big is this? (row counts / `.shape`)

**Question:** Can we compare every Source 1 record to every Source 2 and Source 3 record?

**What we found**

| File | Rows |
|---|---|
| train Source 1 (and ground truth) | 2,206,821 |
| train Source 2 | 5,034,616 |
| train Source 3 | 5,285,603 |
| test Source 1 | 1,732,544 |
| test Source 2 | 4,887,273 |
| test Source 3 | 5,082,316 |

Train brute-force pairs would be about **22.8 trillion**. Test would be about **17.3 trillion**.

**Why it changes the next step:** Phase 3 must **block** (cheaply shortlist plausible matches). A neural net or even Levenshtein on every pair will never finish.

---

## 2. What do the rows actually look like? (`head()` samples)

**Question:** What kinds of mess are in names and addresses — not in theory, in the files?

**What we found (examples)**

- Source 1 is relatively tidy: `Orelee's Barbershop` / `1795 Westchester Drive, High Point, NC`.
- Source 2/3 are noisier: ALL CAPS, Hindi/Kannada/Tamil scripts, junk like `--` or `| www.shivshakti.com`, legal forms in front (`LLC Moncada...`), websites as names (`wilfordhancock.com`), missing addresses (`NaN`).
- Test already shows **France**: `<< Team Ecole` on `175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine`.

**Why it changes the next step:** Phase 2 cleaning is not “make text pretty.” It is “make two spellings of the same company land closer together” (lowercase, strip `Inc`/`Ltd`, expand `St` → `street`). Sample rows told us which rules were worth writing.

---

## 3. How much is missing? (null / blank rates)

**Question:** Can we always use the address? Can we always use the name?

**What we found**

- Source 1 (train and test): **0%** missing names, addresses, and countries.
- Source 2/3: names are almost never missing; **addresses are blank about 2.6–3.4%** of the time (for example 168,967 blank addresses in train Source 2).
- Ground truth: **5.585%** of `matched_entity_ids` cells are empty — those are the **singletons** (see below). IDs are never missing.

**Why it changes the next step:** For those ~3% of S2/S3 rows, matching has to work on **name (and country) alone**. A pipeline that *requires* a street address will silently drop true matches. Features in Phase 4 need a “name-only” path.

---

## 4. Which countries exist? (train vs test)

**Question:** Is `country` a closed list we can one-hot as `{US, India}`?

**What we found**

- Train, all three sources: only **US (~60%)** and **India (~40%)**.
- Test: **India, US, and France (~15% of test Source 1)**.
- Train-only extra countries: none. Test-only: **France**.

**Why it changes the next step:** If the code says `if country in ("US", "India")` and drops the rest, every French Source 1 entity is wrong on the leaderboard. Treat country as **an open string label**. You *may* use “same country?” as a match feature. You may *not* assume the set of countries is finished.

A second, huge finding (from joining ground truth to the sources): **100% of true matches share the same country**. Zero cross-country links in train. So blocking on “same country” is safe *as an equality check*, without listing the country names.

---

## 5. Are IDs unique and well-formed?

**Question:** Could the same `entity_id` appear twice and confuse joins?

**What we found:** every source file has unique IDs, prefixes match the file (`S1-` / `S2-` / `S3-`), and ground truth has exactly one row per train Source 1 ID.

**Why it changes the next step:** We can join on IDs without grouping. Also, an S2 ID is never the true match of two different S1 companies (0 S2/S3 IDs linked to more than one S1). Matching is **one Source 1 → zero or more S2/S3**, not a many-to-many tangle.

---

## 6. Singleton rate — why 5.58% matters for the metric

**Question:** How often does a Source 1 company have *no* partner in S2/S3?

**What we found:** **123,247 / 2,206,821 = 5.58%** singletons. Almost the same in US (5.58%) and India (5.59%).

The contest score is **F₀.₅**, averaged **per Source 1 entity**. For a singleton:

- predict empty list → that entity scores **1.0**
- predict any S2/S3 ID → that entity scores **0.0**

**Why it changes the next step:** A model that “always guesses at least one match” throws away 5.58% of the macro average for free. Phase 6 (threshold tuning) must allow **no match**. Precision-heavy F₀.₅ already punishes false merges; singletons make that even more important.

---

## 7. How many matches does a typical company have?

**Question:** Is this mostly 1-to-1, or do we need to return lists?

**What we found**

- Non-singletons: average **3.67** matches.
- Median (all S1, including singletons): **3**.
- Max: **11** (never 20+). Only **571** entities have 10 or 11 matches.
- Histogram peak: **3 matches** (530,841 entities).
- **80.5%** of S1 rows match *both* Source 2 and Source 3. Some match only S2 (6.5%) or only S3 (7.5%).

**Why it changes the next step:** Output is a **variable-length list**, not a single ID. Blocking must recall several true partners, not the “best one.” High-match entities (10–11) are rare; do not build the whole pipeline around outliers. Because most true lists mix S2 and S3, blocking should search **both** files, not pick a “better source.”

---

## 8. Side-by-side true matches (the most useful EDA)

**Question:** When two rows *are* the same company, how do the strings differ?

This is where Phase 2 came from. A few patterns we actually saw:

| Kind of noise | Real pair (shortened) |
|---|---|
| Legal suffix only on one side | `Ramirez Equity Partners PC` vs `RAMIREZ EQUITY PARTNERS` |
| Suffix moved to the front | `Iconic Management Limited` vs `Limited Iconic Msagemegnt` |
| Honorific + junk | `Ss Finance Private Limited` vs `Sri >> ss finance private limited` |
| Website as the name | `Swamiraj Trading Private Limited` vs `... swamirajtrading.com` |
| Diacritics | `Sandoval Bitcoin` vs `Sandoval Bítcoin` |
| Native script vs Latin | `Ss Finance Private Limited` vs Kannada text |
| Address abbreviation | `910 Pauline Street` vs `910 PAULINE ST` |
| State code vs full name | `... Mumbai City, Maharashtra` vs `... MH` |
| Missing address on a true match | `Sandoval Bitcoin` matched an S3 row with `nan` address |
| House-number typos | `3250 Archer Drive` vs `325 ARCHER DR` (same GT group) |

**Why it changes the next step:** String equality (`name_a == name_b`) will miss most true pairs. We normalize first (Phase 2), then compare with **fuzzy** measures (Jaccard, Levenshtein, TF-IDF — defined in the glossary). We still cannot fix everything with cleaning (a true match named `Mirairi` vs `Saleh, Carissa P., DMD`). Those need address features — and extra caution, because street-only merges create false positives, which F₀.₅ hates.

---

## How to reread this in one pass

1. **Scale** → must block, cannot brute-force.  
2. **France** → country is an open set.  
3. **Same-country true pairs** → block on country equality.  
4. **~5.6% singletons** → empty predictions are part of a good model.  
5. **~3.67 matches, both S2 and S3** → return lists from both sources.  
6. **~3% blank S2/S3 addresses** → name-only matching path.  
7. **True-pair samples** → write the Phase 2 cleaner from evidence, not from a generic “text cleaning” blog post.

The next note (`02_data_cleaning_explained.md`) walks through the actual transformations in `clean_text.py`.

# Glossary — plain-English terms from Phases 1 and 2

These are the words we keep using. Each definition is tied to **this** challenge (three messy business lists, scored with F₀.₅).

---

### Blocking

A **cheap first filter** that says: “these Source 2/3 rows are *plausible* matches for this Source 1 row.” You do **not** run the expensive model on every possible pair (that would be trillions of pairs here). Blocking is allowed to over-include (extra candidates). It must not under-include too much: a true pair that never enters the candidate list can never be predicted. The file `candidate_pairs.tsv` is this shortlist.

---

### Entity resolution (ER)

The task of deciding which records from different lists refer to the **same real-world thing** — here, the same business — when they do not share an ID. Source 1 is the reference list; we attach zero or more Source 2/3 IDs to each Source 1 ID.

---

### EDA (exploratory data analysis)

Looking at the data **before** modeling: sizes, missing values, country mix, how matches are distributed, and real examples of true pairs. Phase 1. The point is to choose the next design (blocking, cleaning rules, “empty list is a valid answer”) from evidence.

---

### F-beta score (here: F₀.₅)

A single number that blends **precision** and **recall**. Beta = 0.5 means **precision is weighted more** (false merges hurt more than missed links). Formula used in the brief:

`F₀.₅ = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)`

It is computed **per Source 1 entity**, then averaged (macro-average). That is why singletons matter: each Source 1 row is one vote, including rows with no true match.

---

### False merge

Saying two records are the same company when they are **not**. In this contest that is a false positive. F₀.₅ punishes this harder than a miss. Example risk: two shops on the same street, different names, if you match on address alone.

---

### Ground truth

The answer key for training: `train_ground_truth.tsv`. Column `matched_entity_ids` is the comma-separated list of correct S2/S3 IDs (empty if none). We never get this file for the official test set; we hold out part of train to score ourselves.

---

### Jaccard similarity

A simple overlap score for two **sets** of tokens (words).

`Jaccard = (words in both) / (words in either)`

Example after cleaning: `{ramirez, equity, partners}` vs `{ramirez, equity, partners}` → 1.0. `{western, madison}` vs `{western}` → 1/2 = 0.5. It ignores word order, which helps with `Associates-Portland` vs `Portland Associates`. It does not understand typos (`tdaaing` vs `trading`).

---

### Normalization (text cleaning)

Rewriting strings into a more standard form so the same company is spelled more similarly: lowercase, strip `Inc`/`Ltd`, expand `St` → `street`, drop `Near Fortis Hospital`, and so on. Phase 2. Not the same as “looking the business up online.”

---

### NFKC

A **Unicode normalization form** (Name: Normalization Form Compatibility Composition). Fancy name, simple job: convert lookalike / compatibility characters toward a standard shape (for example some wide digits or ligatures). Our cleaner runs NFKC, then drops combining accent marks. It does not fix every accented letter (see `Bítcoin` still containing `í`).

---

### Precision vs recall

Imagine you predicted a list of matches for one Source 1 company.

- **Precision:** of the IDs you predicted, what fraction were actually correct? High precision = few false merges.
- **Recall:** of the IDs that were truly correct, what fraction did you find? High recall = few missed matches.

F₀.₅ cares about both, but **leans toward precision**. Blocking is usually tuned for high recall (don’t lose true pairs). The final threshold is tuned for F₀.₅ (don’t over-merge).

---

### Singleton

A Source 1 entity whose ground-truth list is **empty** — no S2/S3 match. About **5.58%** of train Source 1. Predicting any ID for a singleton scores **0.0** on that entity; predicting empty scores **1.0**. “No match” is a first-class answer, not a failure of the pipeline.

---

### TF-IDF

A way to turn a name or address into numbers for cosine similarity (Phase 4 idea, not used yet in cleaning).

- **TF (term frequency):** a word that appears in this record counts more.
- **IDF (inverse document frequency):** a word that appears in *almost every* record (like `mumbai` or `road`) counts less; a rare word (`ramirez`) counts more.

So TF-IDF is “word overlap, but down-weight boring words.” Jaccard treats every token as equal; TF-IDF does not.

---

### Tokenization

Splitting a string into **tokens** (usually words). `"910 PAULINE ST"` → `910`, `pauline`, `st` (and after expansion, `street`). Blocking and Jaccard work on tokens, not on the whole line as one blob. `name_tokens` / `addr_tokens` in the cleaned files are this idea saved to disk (as a sorted unique list when we write the TSV).

---

### Related challenge words (short)

- **Source 1 / 2 / 3:** three independent dumps of businesses. Source 1 is deduplicated; we never predict an S1 ID as a match.
- **Candidate pair:** an (S1, S2 or S3) pair that survived blocking, before the classifier says yes/no.
- **Macro-average:** average the per-entity scores with equal weight per Source 1 row (a company with 11 matches does not count 11 times in the *average*, though its own F₀.₅ still uses its full list).

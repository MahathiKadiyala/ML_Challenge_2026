# ML Challenge 2026: Business Entity Resolution
## My Approach & Methodology (Summary)

---

### 1. What was the problem?
The goal of this competition is **Business Entity Resolution (Record Linkage)**:
- We are given business records from **Source 1 (S1)** and need to find their matching counterpart records across **Source 2 (S2)** and **Source 3 (S3)**.
- The dataset spans three countries: **United States (US)**, **India (India)**, and **France (France)**.
- The evaluation metric is **Macro $F_{0.5}$ score** across all 1,732,544 test entities. Because $F_{0.5}$ values precision twice as much as recall, false merges (linking two different businesses) and false predictions on singletons (businesses with no real match) heavily destroy the score.

---

### 2. Key Discoveries from the Data
Through deep exploratory data analysis on the complete dataset (~12.5 million records), we discovered crucial rules:
1. **Zero Cross-Border Matches:** Exactly 0.00% of true matches cross national borders. We can partition the entire problem strictly by country.
2. **1-to-1 Target Exclusivity:** Every single S2 and S3 entity matches at most **one** S1 entity. Standard independent classifiers create duplicate conflicts; enforcing a 1-to-1 bipartite assignment eliminates false merge collisions.
3. **Cross-Script Indic Text in India:** In Source 1, all company names are written in English (Latin alphabet). In India's Source 2 & Source 3, ~18% of company names are written in native Indic scripts (Hindi/Devanagari, Bengali, Tamil, Telugu, Gujarati, etc.). These are phonetic transliterations (e.g., *"Smart Marketing"* $\leftrightarrow$ *"स्मार्ट मार्केटिंग"*). Without transliteration, standard string matching misses them completely.
4. **House/Street Number Normalization:** 90.2% of real matches with street numbers share the exact same numeric digits once leading zeros and formatting are stripped (e.g., `0201` $\leftrightarrow$ `201`).

---

### 3. How We Built the Solution (Step-by-Step)

#### Step 1: Preprocessing & Transliteration
- Cleaned and stripped legal entity suffixes (`Ltd`, `Pvt`, `Inc`, `LLC`, `SARL`, `SA`) and honorifics (`M/s`, `Shri`).
- Transliterated native Indic scripts into standardized Latin phonetic representations using `unidecode`.
- Extracted and normalized numeric street tokens and website domain stems.

#### Step 2: Multi-Pass Selective Blocking
Comparing all 1.73M S1 records against 9.97M target records would require ~17 trillion comparisons. We designed 5 high-precision blocking passes:
1. **Sorted Core Tokens:** Alphabetically sorted core business words.
2. **First-2 Tokens:** First two distinctive non-stopword tokens.
3. **Street Number + Street Keyword:** Normalized address number combined with street name.
4. **Web Domain Stem / Social Handle:** Domain root names (`metroholding.com` $\leftrightarrow$ `Metro Holding`).
- To prevent memory explosions on common names, every block was capped at `MAX_BLOCK_SIZE = 50`. This yielded an average of ~38 candidates per S1 entity while retaining 88.6%+ true match recall.

#### Step 3: Fast Feature Extraction
For every candidate pair, we computed 18 discriminative features:
- **Name Similarities:** Levenshtein ratio, token sort ratio, token set ratio, core name ratio, exact first-token match, and name length difference.
- **Address Similarities:** Address Levenshtein ratio, address token sort/set ratios, numeric token Jaccard overlap, and exact number match indicator.
- **Structural Indicators:** Domain stem match indicator and data source origin (`S2` vs `S3`).

#### Step 4: Machine Learning Classification (LightGBM)
- Trained a high-capacity **LightGBM Gradient Boosted Decision Tree** model on 3.7 million realistic positive and hard-negative pairs generated from the blocking stages.
- Validated on held-out data to find the optimal decision threshold ($\tau = 0.60$), achieving a validation Macro $F_{0.5}$ of **0.9402** (99.01% precision, 86.76% recall, 96.44% singleton accuracy).

#### Step 5: Greedy 1-to-1 Bipartite Matching
- Rather than accepting all candidates above threshold independently, candidates are sorted by confidence score in descending order.
- Each S2 and S3 target is greedily claimed by at most one S1 entity, strictly enforcing the 1-to-1 constraint and protecting precision.

#### Step 6: Output Assembly & Validation
- Assembled predictions across all three countries for all 1,732,544 test entities.
- Generated the two required submission files:
  1. `matching_results.tsv` (104.6 MB): The final predicted matches.
  2. `candidate_pairs.tsv` (1.91 GB): The complete blocking candidate pool.
- Executed the official `validate_submission.py` to ensure 100% adherence to all competition formatting rules.

---

### 4. Output Files & Results
- **`student_resource/output/matching_results.tsv`**: 1,732,545 lines (1 header + 1,732,544 rows).
- **`student_resource/output/candidate_pairs.tsv`**: 1,732,545 lines (1 header + 1,732,544 rows).
- **`submission.zip`**: The complete submission zip ready for evaluation.

# Amazon ML Challenge 2026: Business Entity Resolution
## Reproduction Instructions

### 1. Environment Setup
Install dependencies using Python 3.10+ (tested on Python 3.13):
```bash
pip install -r requirements.txt
```

### 2. Model Training & Offline Validation
To reproduce training and validation:
```bash
python src/train.py
```
This trains the LightGBM classifier on multi-pass hard negatives and evaluates on the held-out validation benchmark, confirming a Macro $F_{0.5}$ score of **0.9402**.

### 3. Full Test Inference
To run candidate generation, feature extraction, model scoring, and 1-to-1 bipartite assignment across the complete test datasets (France, US, India):
```bash
python src/inference.py
```
This generates the two official competition outputs:
- `output/matching_results.tsv` (Leaderboard submission)
- `output/candidate_pairs.tsv` (Candidate set)

### 4. Format Validation
Validate compliance using the official validator:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

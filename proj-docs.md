# Cat Image Arena — project notes

Midterm analysis for the cat-image preference data. The report question is:

**In a choice between two cat photos, how much is a stable preference for the cat, how much is a preference for side A, and how much is disagreement that does not repeat?**

Two photos of the same cat are a separate, smaller question. The answer so far is that cat identity is the part that replicates. Side A has a small, precise advantage. A second photo of the same cat does not have a reproducible winner.

Prediction of `test.csv` is a consequence of that model, not the thing the model was chosen to maximise.

## How to run

Python 3.12. Package versions are pinned in `requirements.txt`. From the project root:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Use `.venv` as the notebook kernel, and run the notebooks in this order:

1. `notebooks/01_train_eda_cleaning.ipynb` — cleaning, how often cats and photos are shown, and agreement against the cat win-rate gap. Writes `processed/train_cleaned.csv`.
2. `notebooks/02_train_statistical_indices.ipynb` — agreement, side-swap, and reliability. Not required by the model.
3. `notebooks/03_bradley_terry.ipynb` — fits the model, runs the holdouts, and writes the test predictions.

Notebook 3 takes about one minute. Seed is 21.

`dataset/train.csv` and `dataset/test.csv` are left unchanged.

## Data used by the model

Each training row is one response to one ordered pair of photos. There is no respondent id. The six background fields form 69 complete profiles, and the same ordered pair is almost never answered by one profile twice.

Cleaning, in notebook 1:

- The byte-order mark on the id column is removed on read.
- 508 training rows have all six background fields blank together. They stay in the fit. Their rate of choosing A is 0.604, against 0.517 on complete rows, so they are omitted from background breakdowns only.
- Ids, labels, comparison types, and image paths all checked out. No rows were dropped.

| Subset | Rows | What a strength refers to |
| --- | ---: | --- |
| `between_cats` | 21,893 | the cat |
| `within_cat` | 3,003 | the photo, in the comparison model only |
| Test | 6,224 | scored with the rule below |

The 237 cats appear 170 to 245 times, median 212. The 1,296 photos appear 7 to 73 times, median 39. A cat has 3 to 18 training photos, median 5. Twenty-five cats have a photo that never appears in a within-cat pair. Within-cat comparisons are 1,132 unordered pairs.

On the 8,750 between-cat ordered pairs answered twice, both responses match 0.561 of the time. Split by the gap in the two cats' between-cat win rates, that share is 0.512, 0.533, 0.547, 0.567, and 0.647 from the closest fifth of pairs to the furthest. Each cat is shown about 200 times, so one pair barely moves its win rate.

Background fields are not in the model. In notebook 2 every Cramér's V between a background field and `choice` is under 0.05. Pet history, pet care, and pet contact are strongly associated with each other (V of 0.67, 0.51, and 0.48).

## Model

For a between-cat row,

`P(choose A) = logistic(strength(cat A) − strength(cat B) + side)`.

Strengths are fit by L2-penalized logistic regression (`sklearn`, `C = 1`, no extra intercept). The penalty is weak: on the held-out path, `C` of 0.1, 1, 10, and 100 all give a between-cat log loss within 0.002 of each other. Strengths are centered, so an average cat has strength 0. A cat that is absent from the fit is given strength 0.

`side` is the log-odds of choosing A when the two cats have equal strength. It is not the same object as a photo feature.

The same equation with a strength per photo was fit as a competitor, once on between-cat rows and once on within-cat rows. It is not used for the submitted probabilities. On within-cat test rows the prediction is the within-cat training rate of choosing A, which is 0.514.

## Validation

Pair folds keep both side orders of a comparison on the same side of the split. Five folds. The baseline for a fold is the rate of choosing A in that fold's fit rows.

A second check removes a quarter of the cats and, among pairs of the cats that remain, holds out a fifth of the pairs. That produces three scores: both cats seen but the pair unseen, one cat unseen, and both cats unseen. It is repeated five times.

The side-effect interval resamples unordered pairs with replacement (200 draws) and refits.

Log loss is the score used to compare models. Accuracy is reported with it because a constant prediction of A already scores 0.520 on between-cat rows. A photo model can raise accuracy while making the probabilities worse. That happens for within-cat photos at `C = 1`: accuracy rises to 0.543 and log loss gets worse (0.703 against a side-only 0.693).

## Results

Full between-cat fit, 237 cats:

| Quantity | Value |
| --- | ---: |
| Side effect, log-odds | 0.088 |
| P(choose A) when the cats are equal | 0.522 |
| Pair-bootstrap 95% interval for that probability | 0.516 to 0.529 |
| SD of cat strength | 0.548 |
| SD of strength(cat A) − strength(cat B) | 0.668 |
| Share of fitted rows whose cat gap exceeds the side effect | 87.8% |

The last share describes the fitted strengths. It is not a hit rate on new pairs. The holdout scores are the evidence.

Five-fold pair holdout:

| Model | Log loss | Brier | Accuracy |
| --- | ---: | ---: | ---: |
| Between, side only | 0.6924 | 0.2496 | 0.520 |
| Between, photo strength + side, C = 1 | 0.6675 | 0.2367 | 0.603 |
| Between, cat strength + side, C = 1 | 0.6552 | 0.2316 | 0.608 |
| Within, side only | 0.6933 | 0.2501 | 0.514 |
| Within, photo strength + side, C = 0.1 | 0.6901 | 0.2485 | 0.536 |
| Within, photo strength + side, C = 1 | 0.7025 | 0.2538 | 0.543 |

Cat strength beats the side-only rate. Photo strength beats it by less, and on within-cat rows the best penalty on the grid (`C = 0.1`) gains only 0.003 log loss. Weaker penalties overfit. Within-cat test rows therefore get the side rate and no photo strength.

Cat holdout, mean of five repeats:

| Slice | Mean rows | Model log loss | Side-only log loss |
| --- | ---: | ---: | ---: |
| Both cats seen, pair held out | 2,467 | 0.662 | 0.692 |
| One cat unseen | 8,250 | 0.685 | 0.693 |
| Both cats unseen | 1,322 | 0.694 | 0.694 |

The gain is about 0.030 when both cats were in the fit, about 0.007 when one was not, and zero when neither was. Standard deviations of the model log loss across the five repeats are 0.007, 0.004, and 0.001.

This matches notebook 2, and the repeated-pair gradient in notebook 1. Split-half correlation of side-adjusted strength was about 0.69 for cats and about 0.15 for within-cat photos. Cohen's kappa on a repeated ordered pair was 0.11 overall and 0.05 within a cat. Agreement on a repeated between-cat pair rises from 0.51 to 0.65 as the cats' win rates move apart.

## Test predictions

`processed/test_predictions.csv` has the original test columns. `prob_a` is filled, and `choice` is A when `prob_a ≥ 0.5`, otherwise B. Counts: 4,317 A and 1,907 B.

| Test slice | Rows | Mean `prob_a` | Share predicted A |
| --- | ---: | ---: | ---: |
| Between, both cats seen in training | 2,743 | 0.523 | 0.565 |
| Between, one cat unseen | 1,556 | 0.520 | 0.542 |
| Between, both cats unseen | 926 | 0.522 | 1.000 |
| Within cat | 999 | 0.514 | 1.000 |

The last two slices are constant probabilities above one half, so the decision rule labels every one of them A. Of the 999 within-cat test rows, 630 contain a photo that never appears in training. Those rows would have fallen back to the average photo even if photo strengths had been used.

The original test labels are blank, so these slices are not scored. The cat-holdout table is the training analogue.

## What was not fit

A deep model on the pixels was not used. Within a cat, the photo-level target barely replicates, so a network would be fitting disagreement. Between cats, the stable signal is which cat, and a photo embedding would mostly rediscover that identity. Readable visual attributes of the cat are still an optional later check against the 237 strengths in `processed/cat_strengths.csv`. They are not required for the result above.

## Files

| Path | Role |
| --- | --- |
| `requirements.txt` | Pinned packages for Python 3.12 |
| `notebooks/01_train_eda_cleaning.ipynb` | Cleaning, appearance counts, agreement by win-rate gap |
| `notebooks/02_train_statistical_indices.ipynb` | Kappa, side swap, reliability, Cramér's V |
| `notebooks/03_bradley_terry.ipynb` | Model, holdouts, test file |
| `processed/train_cleaned.csv` | Training table used by notebooks 2 and 3 |
| `processed/cat_strengths.csv` | 237 centered cat strengths and win counts |
| `processed/test_predictions.csv` | Test file with `prob_a` and `choice` filled |
| `outputs/regularization_path.csv` | Held-out scores across `C` |
| `outputs/pair_holdout_logloss.csv` | Primary comparison |
| `outputs/cat_holdout_logloss.csv` | Scores by whether the cat was seen |
| `outputs/side_effect_bootstrap.csv` | Side-effect interval |
| `outputs/figures/` | Figures cited by the notebooks |

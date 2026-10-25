# Cat Image Arena

Pairwise cat-image preferences, and the midterm analysis of those choices. The analysis notes are in `proj-docs.md`.

## Setup

Python 3.12. From the project root:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Use `.venv` as the Jupyter kernel, then run the notebooks in order:

1. `notebooks/01_train_eda_cleaning.ipynb`
2. `notebooks/02_train_statistical_indices.ipynb`
3. `notebooks/03_bradley_terry.ipynb`

`dataset/train.csv` and `dataset/test.csv` are left unchanged.

## Dataset

A dataset of pairwise cat-image preferences for data analysis and prediction.

## Objectives

- **Explore and interpret the data:** examine human preferences, disagreement between annotators, and relationships between annotator characteristics and image preferences.
- **Predict preferences:** use the training data to predict each annotator’s image preference in the test data. In `test.csv`, fill in `prob_a` with the predicted probability of choosing image A (between 0 and 1). Fill in `choice` with `A` if `prob_a ≥ 0.5`, or `B` otherwise.

Each row represents one annotator's response to one image pair. A pair may have multiple responses from different annotators.

## Files

| File | Description |
| --- | --- |
| `dataset/train.csv` | 24,896 responses with observed `choice` values. |
| `dataset/test.csv` | 6,224 test rows with `choice` and `prob_a` left blank. |
| `dataset/images/` | 1,558 images organized into cat folders. |

## Image paths

Get the images from [Hugging Face](https://huggingface.co/datasets/missing12/cat-image-arena-images/tree/main/images).

The `a_image` and `b_image` fields contain paths relative to `dataset/images/`. For example:

```text
Table value: cat_000002/cat_000002_0001.jpg
Local file:  dataset/images/cat_000002/cat_000002_0001.jpg
```

## Fields

| Field | Meaning |
| --- | --- |
| `train_row_id` | Unique training-row identifier; present only in the training file. |
| `test_row_id` | Unique test-row identifier; present only in the test files. |
| `a_cat`, `b_cat` | Cat identifiers for sides A and B. |
| `a_image`, `b_image` | Image paths for sides A and B. |
| `choice` | Observed preference: `A` or `B`. Withheld in the test files. |
| `prob_a` | Predicted probability of choosing image A, between 0 and 1; present only in the test file. |
| `comparison_type` | `within_cat`: different photos of the same cat; `between_cats`: photos of different cats. |
| `gender` | Self-reported gender. |
| `pet_history` | Household pet ownership: `current`, `past`, or `never`. |
| `pet_care` | Personal pet-care experience: `never`, `occasionally`, `past_regular`, or `current_regular`. |
| `pet_contact` | Frequency of interaction with pets: `never`, `rarely`, `sometimes`, `often`, or `daily`. |
| `future_pet` | Intention to own or continue owning pets: `yes`, `no`, or `unsure`. |
| `mbti_ie` | `I` or `E`, derived from self-reported MBTI; `unknown` means unsure or not tested. |

For research and educational use only.

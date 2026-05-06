# TheEnvelope

Machine learning pipeline that predicts whether a movie will receive an Oscar nomination, using historical film data from TMDb, IMDb, and the Academy Awards

**Authors: Vaishvi Patel & Himani Mehta**

---

## Overview

Each year, thousands of movies are released, yet only a small percentage receive nominations from major award organizations. By analyzing historical data from movies, including ratings, box office revenue, genre, release timing, critic reviews, and production details, we aim to identify patterns that influence award nominations.

---

## Data Sources

| Source | What it provides |
|--------|-----------------|
| [TMDb API](https://www.themoviedb.org/documentation/api) | Title, release date, genres, runtime, budget, revenue, cast, director, MPAA rating |
| [IMDb Datasets](https://developer.imdb.com/non-commercial-datasets/) | Audience ratings, vote counts, genre classifications |
| [Kaggle — The Oscar Award](https://www.kaggle.com/datasets/unanimad/the-oscar-award) | Oscar nomination and win records |

---

## How to Run

### 1. Install libraries

pip install -r requirements.txt

### 2. Collect data

python src/data_collection.py

Downloads IMDb files, Oscars CSV, and fetches movie metadata from the TMDb API for 2010–2024. Saves merged dataset to `data/merged/merged_dataset.csv`

### 3. Run ETL to clean and transform

python src/etl.py

Filters to feature films, handles missing budget/revenue using year-based median, removes duplicates, and standardizes column types

### 4. Engineer features

python src/features.py

Builds the feature dataset used for modeling, including:
- Log-scaled budget, revenue, and vote counts
- Profit margin and ROI
- Awards season flag and release quarter
- One-hot genre encoding and genre nomination rate
- Ordinal MPAA encoding
- Director and cast prior nomination counts
- Combined mean rating from TMDb and IMDb

### 5. Train and evaluate

Open and run `model/modeling.ipynb` (the notebook trains a Logistic Regression baseline and a Random Forest classifier, evaluating both with ROC-AUC, precision, recall, and F1-score)

---

## Models

| Model | Notes |
|-------|-------|
| Logistic Regression | Interpretable baseline with `class_weight="balanced"` |
| Random Forest | Primary model (handles nonlinear relationships and class imbalance) |

ROC-AUC is the primary evaluation metric due to significant class imbalance (nominated films are a small fraction of all releases)

---

## Limitations

- Oscar nominations only (Golden Globes and other awards are not included)
- Nomination outcomes are partially subjective (studio campaigns and industry relationships are unquantifiable)
- `genre_nom_rate` is computed across the full dataset (may introduce minor leakage)
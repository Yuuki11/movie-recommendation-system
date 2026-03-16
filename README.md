# Movie Recommendation System

This is a movie recommendation web app built with Flask, Pandas, NumPy, and
scikit-learn. A user selects the genres they like, sets a minimum IMDb rating,
and the app returns movie suggestions ranked with a logistic regression model.

## What It Does

- lets users choose genres they like or want to avoid
- predicts movie matches using logistic regression
- uses a simple train/test split for the model
- supports a Kaggle movie dataset with a sample-data fallback
- includes a small frontend built with HTML, CSS, and JavaScript

## Dataset

The main dataset used for this project is
`rounakbanik/the-movies-dataset` from Kaggle.

The app reads:

- `movies_metadata.csv`
- `links_small.csv`
- `ratings_small.csv`

If those files are not available in `data/kaggle/`, the app falls back to the
sample dataset in `data/movies.csv`.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

## Dependencies

`requirements.txt` contains the versions that run cleanly on this machine.

`requirements-legacy.txt` keeps the older dependency set that matches the
original version of the project more closely.

## Notes

- the model is intentionally simple and easy to follow
- downloaded Kaggle files and generated data are ignored from git

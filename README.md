# Movie Recommendation System

Flask-based movie recommendation web app built with Python, Pandas, NumPy, and
scikit-learn. The project uses logistic regression to model user preferences
from selected genres and return ranked movie suggestions.

## Stack

- Python 3.8.7
- Flask 1.1.2
- pandas 1.1.5
- NumPy 1.19.4
- scikit-learn 0.23.2

## Features

- Interactive genre-based recommendation form
- Logistic regression recommendation pipeline
- Kaggle dataset support with sample-data fallback
- HTML, CSS, and JavaScript frontend

## Dataset

Primary dataset: `rounakbanik/the-movies-dataset` from Kaggle.

The app uses:

- `movies_metadata.csv`
- `links_small.csv`
- `ratings_small.csv`

If those files are not available in `data/kaggle/`, the app falls back to the
bundled sample file in `data/movies.csv`.

## Run

```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Notes

- The code is written to reflect a December 2020-era Python stack.
- Generated data and downloaded Kaggle files are ignored from git.

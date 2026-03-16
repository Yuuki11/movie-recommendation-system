# Movie Recommendation System

Beginner-friendly Flask movie recommendation project built with Python, Pandas,
NumPy, and scikit-learn. It uses logistic regression, genre-based features, and
a simple train/test split to predict which movies best match a user's selected
preferences.

## Original 2020 Stack

- Python 3.8.7
- Flask 1.1.2
- pandas 1.1.5
- NumPy 1.19.4
- scikit-learn 0.23.2

These original course-era versions are preserved in
`requirements-2020.txt`.

## Local Run Stack

For this Apple Silicon Mac, the runnable local environment uses a compatibility
stack in `requirements.txt`:

- Python 3.9+ or Python 3.13
- Flask 3.1.3
- pandas 3.0.1
- NumPy 2.4.3
- scikit-learn 1.8.0

## Features

- Interactive genre-based recommendation form
- Logistic regression recommendation pipeline
- Simple 80/20 train-test workflow
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

## Run On This Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## 2020 Environment Note

If you want the exact 2020 package versions, use `requirements-2020.txt` with
an older Python environment such as Python 3.8 on an Intel/x86_64 machine,
Linux VM, or container. Those exact scientific package versions do not install
cleanly on this modern arm64 macOS system.

## Notes

- The code is written to reflect a December 2020-era Python stack.
- The modeling approach is intentionally simple to match a beginner ML course project.
- `requirements.txt` is for local compatibility.
- `requirements-2020.txt` preserves the original project-era dependency versions.
- Generated data and downloaded Kaggle files are ignored from git.

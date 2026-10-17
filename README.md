# YuvaIntern Week 4 — AI Project Deployment & Capstone

## Project: Wine Classifier Prediction API

An end-to-end machine-learning application that classifies a wine sample into one of the three **dataset cultivar categories** using 13 numeric chemical measurements. It demonstrates data preparation, candidate-model comparison, held-out testing, model serialization with Joblib, and a FastAPI prediction endpoint.

> **Scope note:** the labels are cultivar categories from the educational Wine Recognition dataset. They are not wine quality grades and the API does not assess safety, authenticity, or suitability for consumption.

## Features

- Reproducible 80/20 stratified train/test split.
- Compares Logistic Regression and Random Forest using 5-fold stratified cross-validation.
- Reports held-out accuracy, macro precision, macro recall, macro F1, class report, and confusion matrix.
- Saves a fitted preprocessing/model pipeline with `joblib`.
- FastAPI routes: `GET /`, `GET /health`, `GET /model-info`, and `POST /predict`.
- Pydantic input validation; invalid or missing features receive a validation error.
- Automated smoke tests with `pytest`.
- Includes Docker and Render configuration as deployment options.

## Project structure

```text
app/                 FastAPI app and request/response schemas
ml/train.py          Dataset export, preprocessing, model comparison, evaluation, serialization
tests/test_api.py    API smoke tests
artifacts/           Serialized model, metadata, metrics, example request, reports
data/wine.csv        Dataset exported reproducibly by the training script
figures/             Model-comparison and confusion-matrix plots
report/              Internship report (.docx)
presentation/        Project presentation (.pptx)
```

## Run locally on Windows PowerShell

Use Python 3.11 or 3.12 for a predictable deployment environment. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ml.train
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation and `http://127.0.0.1:8000/health` to verify model loading. The included `artifacts/sample_request.json` is a valid example body for `POST /predict`.

Run automated tests in another terminal while the environment is active:

```powershell
pytest -q
```

## Example request

Use the full JSON body from `artifacts/sample_request.json` in `POST /predict`. The API expects all 13 fields: `alcohol`, `malic_acid`, `ash`, `alcalinity_of_ash`, `magnesium`, `total_phenols`, `flavanoids`, `nonflavanoid_phenols`, `proanthocyanins`, `color_intensity`, `hue`, `od280_od315_of_diluted_wines`, and `proline`.

A successful response contains the predicted numeric class, its dataset class label, probability estimates by class, and the selected model name. Predictions are based on the saved model and the model pipeline's own preprocessing.

## Deployment options

### Docker

```bash
docker build -t wine-classifier-api .
docker run --rm -p 8000:8000 wine-classifier-api
```

### Render

Push this project to a GitHub repository, create a new Render Web Service connected to that repository, and use the included `render.yaml` configuration (or set build command to `pip install -r requirements.txt && python -m ml.train` and start command to `uvicorn app.main:app --host 0.0.0.0 --port $PORT`). After a successful deployment, verify `/health` and `/docs` using the public service URL. This repository is deployment-ready but is not claimed to be publicly hosted until a deployment provider returns a live URL.

## Reproducibility and limitations

The scikit-learn Wine Recognition dataset is bundled with scikit-learn, so the training script does not require a separate dataset download. The fixed random seed makes the split and model comparisons reproducible within compatible library versions. This compact educational dataset is useful for a deployment demonstration but should not be treated as evidence that the classifier will generalize to arbitrary real-world samples.

## Dataset reference

Scikit-learn, *Wine recognition dataset*: https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset

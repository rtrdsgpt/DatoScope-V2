# DatoScope V2

DatoScope is a multipage Streamlit application for synthetic data generation, dataset upload, preprocessing, exploratory data analysis, supervised learning, clustering, and model comparison.

This is a solo continuation of a group course project originally built for the MA5755 course
(IIT Madras) by Tanmoy (MA25M026), Mehak (MA25M016), and Aritra (MA25M005), under the supervision
of Prof. Rakhi Singh. V2 extends the original classical-ML/EDA app with an ETL pipeline, API layer,
MLOps, cloud/Kubernetes deployment, and an AI co-pilot layer — see `todo.md` for the roadmap.

#### (Now supporting Light and Dark Modes!)

## Highlights

- Generate datasets inside the app for regression, classification, or clustering
- Upload a single dataset or separate train/test files
- Optionally create a train/test split during dataset generation
- Clean train and test data with missing-value handling, outlier removal, duplicate removal, scaling, and categorical encoding
- Run EDA with summary statistics, distributions, Q-Q plots, correlations, scatter plots, and explainable variance ranking
- Train regression models: Linear Regression, Ridge, Lasso
- Train classification models: Logistic Regression, Random Forest, KNN
- Visualize individual trees from trained Random Forest classifiers
- Run clustering models: K-Means, DBSCAN, Hierarchical Clustering
- Compare regression, classification, and clustering results in dedicated pages
- Download trained supervised models as `.pkl`

## Data Input Modes

The sidebar supports three workflows:

1. `Generate Dataset`
   Create synthetic regression, classification, or clustering datasets with controls for:
   - dataset type
   - sample count
   - noise
   - number of clusters / arms
   - number of features
   - target column name
   - optional generated train/test split
   - random seed

2. `Upload Single File`
   Upload one file and let the app create an internal train/test split during supervised modeling.
   Uploaded datasets can also use categorical encoding during preprocessing.

3. `Upload Train/Test`
   Upload a train file and an optional test file.
   If the test file is present, the app uses it directly instead of creating a split.

## Pages

- `app.py`
  Streamlit entry point; builds the sidebar navigation for the pages below
- `pages/1_EDA.py`
  Exploratory data analysis
- `pages/2_Supervised_Modeling.py`
  Regression and classification workflows, random forest controls, and tree visualization
- `pages/3_Clustering.py`
  Clustering workflows, cluster-size plots, and optional ground-truth metrics
- `pages/4_Comparison.py`
  Model comparison dashboard with improved regression scoring and explicit winner-selection logic

## Project Structure

```text
DatoScope/
├── app.py
├── data_loader.py
├── pages/
│   ├── 1_EDA.py
│   ├── 2_Supervised_Modeling.py
│   ├── 3_Clustering.py
│   └── 4_Comparison.py
├── utils/
│   ├── app_state.py
│   ├── data_input.py
│   ├── generators.py
│   ├── io.py
│   ├── modeling.py
│   ├── preprocessing.py
│   └── ui.py
├── scripts/
│   ├── 01_generate_data.py
│   ├── 02_clean_data.py
│   ├── 03_eda.py
│   └── 04_visualization.py
├── train.py
└── requirements.txt
```

## Setup

Use Python 3.11 for the most reliable dependency compatibility.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run The App

```bash
streamlit run app.py
```

## Run The Offline Pipeline

```bash
python train.py
```

This pipeline runs:

1. synthetic dataset generation
2. cleaning and preprocessing
3. EDA reporting
4. static plot generation

## Supported File Types

| Extension | Notes |
|---|---|
| `.csv` | Standard CSV upload |
| `.xlsx` / `.xls` | Excel upload with simple header detection |
| `.zip` | Must contain one CSV |
| `.data` | Headerless comma- or whitespace-delimited files |

## Notes

- Generated datasets can now be created specifically for regression, classification, or clustering.
- The sidebar includes a selected ML task control so the interface can stay focused on one task at a time.
- Uploaded datasets can encode categorical variables using One-Hot or Label encoding during preprocessing.
- Classification is inferred from the selected target column but can also be chosen manually in the supervised modeling page.
- Classification train/test splitting now falls back safely when a class has too few samples for strict stratification.
- Model export currently supports supervised models.
- Clustering runs on the train dataset only when a separate test file is present.
- If a dataset contains ground-truth `label` values, clustering comparison can also report Fowlkes-Mallows and Rand Index scores.
- Regression comparison now considers both predictive quality and generalization instead of choosing winners from raw test R² alone.
- Clustering comparison now selects the best algorithm by counting how many of the five tracked clustering metrics each model wins.

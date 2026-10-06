# Prediction of Mechanical Properties of Forged Steels using Machine Learning

## Objective

Build machine-learning models that predict the mechanical properties of forged
steel parts, namely **Ultimate Tensile Strength (UTS)**, **Yield Strength (YS)**
and **Hardness**, from:

- **Chemical composition** (e.g. C, Mn, Si, Cr, Ni, Mo, ... in wt%)
- **Forging / processing parameters** (where available in the dataset)

Goals of the project:

1. Learn the full materials-informatics workflow (data exploration, cleaning,
   feature engineering, modelling, validation, interpretation).
2. Quantify how well composition (and process) alone can predict properties.
3. Use model interpretation (feature importance, SHAP) to check whether the
   learned trends agree with known physical metallurgy (e.g. carbon and
   hardenability effects, solid-solution strengthening).

## Dataset

- **Source:** public GitHub repository by *saikirananugam* (add the exact
  repository URL and file name here once you have downloaded it).
- **Size:** 63,162 rows x 23 columns.
- **Content:** chemical elements and mechanical properties of steels.
- **Location in this project:** `data/forged_steel.csv`
  (rename the downloaded file, or edit `DATA_PATH` in the scripts).

> Note: before modelling, confirm in `01_data_exploration.py` which columns are
> inputs (composition, process) and which are targets (UTS, YS, hardness).
> Also check the repository's license and the origin of the data, and record
> them here.

## Methodology

1. **Data exploration (EDA)**: shape, dtypes, missing values, descriptive
   statistics, distributions, correlation heatmap. *(`01_data_exploration.py`)*
2. **Data cleaning**: handle missing values, duplicates, physically impossible
   values and outliers (use metallurgical judgement, not only statistics).
3. **Feature engineering**: optional derived features such as carbon
   equivalent (CE), alloying-sum features, ratios.
4. **Train / test split**: hold out a test set; use cross-validation for model
   selection. Consider grouped splits if the same alloy/heat appears many times
   (to avoid data leakage).
5. **Baseline models**: Linear / Ridge regression.
6. **Advanced models**: Random Forest, Gradient Boosting, (optionally XGBoost
   and a small neural network).
7. **Evaluation**: R^2, MAE, RMSE per target; parity plots (predicted vs.
   measured); residual analysis.
8. **Interpretation**: permutation importance and SHAP values, compared with
   known metallurgy.
9. **Reporting**: summarise results, limitations and next steps.

## Project structure

```
.
├── README.md
├── requirements.txt
├── 01_data_exploration.py
├── data/
│   └── forged_steel.csv      # you add this file
└── figures/                  # created automatically by the scripts
```

## Installation

```bash
# 1. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Put the dataset at data/forged_steel.csv, then run:
python 01_data_exploration.py

# Optionally pass a different path:
python 01_data_exploration.py path/to/your_file.csv
```

Plots are saved in the `figures/` folder. To work interactively, start Jupyter:

```bash
jupyter notebook
```

## License and data attribution

Add your own license here, and credit the original dataset authors/repository.

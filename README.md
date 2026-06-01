# 🐍 Python Hands-on Projects

Applied Machine Learning and Python projects completed as part of the **Applied AI Solutions Development** program at **George Brown College**.

---

## 📁 Structure

```
Python-Hands-on-Projects/
│
├── ml-tasks/
│   ├── Preprocessing_SimpleImputer.ipynb
│   ├── Preprocessing_IterativeImputer.ipynb
│   ├── Covid-19_data_analysis.ipynb
│   ├── House_price_prediction__Linear_Regression_.ipynb
│   ├── AutoFeatureSelector_tool.py
│   ├── RandomForest_hyperparameter_tuning.ipynb
│   ├── Customer_classification__ensemble_ML_.py
│   └── combined_project_models_v2.ipynb
│
└── web-scraping/
    └── Web_Scraping.py
```

---

## 🗂️ Projects

### 🤖 ML Tasks

| File | Description | Key Techniques |
|------|-------------|----------------|
| `Preprocessing_SimpleImputer` | Missing value imputation — univariate | `SimpleImputer`, mean/median strategy |
| `Preprocessing_IterativeImputer` | Missing value imputation — multivariate | `IterativeImputer`, feature-based estimation |
| `Covid-19_data_analysis` | Global COVID-19 spread analysis | `pandas` groupby, EDA |
| `House_price_prediction` | Real estate price prediction | `LinearRegression`, R² score |
| `AutoFeatureSelector_tool` | Automated multi-method feature selection tool | Pearson, Chi-Square, RFE, Lasso, Random Forest, LightGBM |
| `RandomForest_hyperparameter_tuning` | Random Forest optimization | `RandomizedSearchCV`, ROC AUC, Confusion Matrix |
| `Customer_classification__ensemble_ML_` | Customer binary classification pipeline | Random Forest, SVM, Logistic Regression, Voting Ensemble |
| `Penguin_species_classification` | Penguin species classification | Multiple classifiers on Palmer Penguins dataset |

---

### 🌐 Web Scraping

| File | Description | Key Techniques |
|------|-------------|----------------|
| `Web_Scraping` | Yelp restaurant review scraper | Selenium, ChromeDriver, CSV export |

> Requires Google Chrome installed locally. Extracts reviewer name, star rating, and full review text from any Yelp restaurant page.

---

## 🛠️ Tech Stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `lightgbm` · `selenium`

---

## ⚙️ Setup

```bash
git clone https://github.com/SergeyMolkov/Python-Hands-on-Projects.git
cd Python-Hands-on-Projects
pip install -r requirements.txt
```

> **Notes:**
> - `Web_Scraping.py` requires Google Chrome installed locally
> - `AutoFeatureSelector_tool.py` requires `fifa19.csv` — available on [Kaggle](https://www.kaggle.com/karangadiya/fifa19)
> - `RandomForest_hyperparameter_tuning.ipynb` requires `data/2015.csv` — available on [Kaggle](https://www.kaggle.com/cdc/behavioral-risk-factor-surveillance-system)

---

## 👤 Author

**Sergey Molkov**
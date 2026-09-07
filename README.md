# 🐍 Python Hands-on Projects

Applied Machine Learning and Python projects completed as part of the **Applied AI Solutions Development** program at **George Brown College**.

---

## 📁 Structure

```
Python-Hands-on-Projects/
│
├── classic-ml-tasks/
│   ├── Preprocessing_SimpleImputer.ipynb
│   ├── Preprocessing_IterativeImputer.ipynb
│   ├── Covid-19_data_analysis.ipynb
│   ├── House_price_prediction__Linear_Regression_.ipynb
│   ├── AutoFeatureSelector_tool.py
│   ├── RandomForest_hyperparameter_tuning.ipynb
│   ├── Customer_classification__ensemble_ML_.py
│   └── Penguin_species_classification.ipynb
│
├── text-picture-video-sound/
│   ├── Text2Voice_AudioBook.ipynb
│   ├── Text_Summarization.ipynb
│   ├── Picture_Object_Extraction_from_Background.ipynb
│   ├── Picture_Object_Extraction_Kittens.jpg
│   ├── Video_Cartoon_Filter.ipynb
│   ├── Video_Cartoon_Filter_Mickey_face.png
│   └── Video_Thief_Detector.ipynb
│
└── web-scraping/
    └── Web_Scraping.py
```

---

## 🗂️ Projects

### 🤖 Classic ML Tasks

| File                                      | Description                                   | Key Techniques                                           |
| ----------------------------------------- | --------------------------------------------- | -------------------------------------------------------- |
| `Preprocessing_SimpleImputer`           | Missing value imputation — univariate        | `SimpleImputer`, mean/median strategy                  |
| `Preprocessing_IterativeImputer`        | Missing value imputation — multivariate      | `IterativeImputer`, feature-based estimation           |
| `Covid-19_data_analysis`                | Global COVID-19 spread analysis               | `pandas` groupby, EDA                                  |
| `House_price_prediction`                | Real estate price prediction                  | `LinearRegression`, R² score                          |
| `AutoFeatureSelector_tool`              | Automated multi-method feature selection tool | Pearson, Chi-Square, RFE, Lasso, Random Forest, LightGBM |
| `RandomForest_hyperparameter_tuning`    | Random Forest optimization                    | `RandomizedSearchCV`, ROC AUC, Confusion Matrix        |
| `Customer_classification__ensemble_ML_` | Customer binary classification pipeline       | Random Forest, SVM, Logistic Regression, Voting Ensemble |
| `Penguin_species_classification`        | Penguin species classification                | Multiple classifiers on Palmer Penguins dataset          |

---

### 🎬 Text, Picture, Video & Sound

| File                                          | Description                                        | Key Techniques                                                                  |
| --------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------- |
| `Text2Voice_AudioBook`                      | Turns a PDF book into a narrated MP3 audiobook     | `PyPDF2`, regex text cleaning, `pyttsx3` (SAPI5 TTS)                        |
| `Text_Summarization`                        | Extractive summarization of a Medium article       | Selenium +`BeautifulSoup`, `sumy` (TextRank, LexRank, Luhn, LSA), `summa` |
| `Picture_Object_Extraction_from_Background` | Foreground extraction via thresholding             | Simple / Otsu / Adaptive thresholding, morphological ops, contours              |
| `Video_Cartoon_Filter`                      | Live webcam filter that masks faces with a cartoon | Haar cascades, alpha-channel overlay,`VideoWriter`                            |
| `Video_Thief_Detector`                      | Motion-detection alarm for a monitored scene       | Background subtraction (`cv2.absdiff`), contour area filtering                |

**Notes on this section:**

- `Text2Voice_AudioBook` expects a PDF placed in a local `Book/` folder; the cleaning step normalizes typographic quotes and dashes so the synthesized speech pauses naturally. Voice output depends on the SAPI5 voices installed on the machine (Windows).
- `Text_Summarization` renders the article with Selenium first, since Medium loads body text via JavaScript. Requires Google Chrome.
- `Picture_Object_Extraction_from_Background` uses `Picture_Object_Extraction_Kittens.jpg`. Conclusion of the experiment: a manually tuned threshold (T=240) beat Otsu on this photo, because the near-white background breaks Otsu's bimodal assumption.
- `Video_Cartoon_Filter` and `Video_Thief_Detector` both need a working webcam and write their output to `.avi`. The cartoon filter reads `Video_Cartoon_Filter_Mickey_face.png` — swap in any RGBA image to change the mask.

---

### 🌐 Web Scraping

| File             | Description                    | Key Techniques                     |
| ---------------- | ------------------------------ | ---------------------------------- |
| `Web_Scraping` | Yelp restaurant review scraper | Selenium, ChromeDriver, CSV export |

> Requires Google Chrome installed locally. Extracts reviewer name, star rating, and full review text from any Yelp restaurant page.

---

## 🛠️ Tech Stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `lightgbm` · `selenium` · `opencv-contrib-python` · `PyPDF2` · `pyttsx3` · `sumy` · `summa` · `nltk` · `beautifulsoup4`

---

## ⚙️ Setup

```bash
git clone https://github.com/SergeyMolkov/Python-Hands-on-Projects.git
cd Python-Hands-on-Projects
pip install -r requirements.txt
```

> **Notes:**
>
> - `Web_Scraping.py` and `Text_Summarization.ipynb` require Google Chrome installed locally
> - `Video_Cartoon_Filter.ipynb` and `Video_Thief_Detector.ipynb` require a webcam; the cartoon filter also downloads Haar cascade XMLs into a local `cascades/` folder on first run
> - `Text2Voice_AudioBook.ipynb` requires TTS voices installed in the OS (`pyttsx3` uses SAPI5 on Windows, NSSpeechSynthesizer on macOS, espeak on Linux)
> - `AutoFeatureSelector_tool.py` requires `fifa19.csv` — available on [Kaggle](https://www.kaggle.com/karangadiya/fifa19)
> - `RandomForest_hyperparameter_tuning.ipynb` requires `data/2015.csv` — available on [Kaggle](https://www.kaggle.com/cdc/behavioral-risk-factor-surveillance-system)

---

## 👤 Author

**Sergey Molkov**

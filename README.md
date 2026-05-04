# SpamShield AI — Spam SMS Detection

###

---

## Overview

SpamShield AI is a fully optimized, end-to-end Machine Learning pipeline that classifies SMS messages as **Spam** or **Ham (Legitimate)** using Natural Language Processing. The system achieves **98.92% accuracy** and a **95.97% F1-Score** on 5,572 real-world messages from the SMS Spam Collection dataset.

Three classifiers — Naïve Bayes, Logistic Regression, and Support Vector Machine — are benchmarked against each other using TF-IDF vectorization with bigram features. The SVM (LinearSVC) emerges as the best model across all metrics.

---

## Project Structure

```
spam_sms_detection/
├── notebook/
│   └── Spam_SMS_Detection.ipynb      # Full analysis notebook with pre-loaded outputs
├── ui/
│   └── index.html                    # Million-dollar interactive web UI
├── assets/
│   ├── flowchart.png                 # End-to-end ML pipeline flowchart
│   ├── fig1_eda.png                  # EDA: class balance, length, word count
│   ├── fig2_features.png             # Feature prevalence & correlation heatmap
│   ├── fig3_model_comparison.png     # Side-by-side model performance comparison
│   ├── fig4_confusion_matrices.png   # Confusion matrices for all 3 models
│   ├── fig5_roc.png                  # ROC curves with AUC scores
│   └── fig6_cv.png                   # 5-fold cross-validation F1 scores
├── spam.csv                          # SMS Spam Collection dataset
└── README.md                         # This file
```

---

## Dataset

| Property       | Value                        |
| -------------- | ---------------------------- |
| Source         | SMS Spam Collection (UCI ML) |
| Total Messages | 5,572                        |
| Ham (Legit)    | 4,825 (86.6%)                |
| Spam           | 747 (13.4%)                  |
| Features Used  | Raw SMS text                 |

**Key Observations from EDA:**

- Spam messages are on average **2× longer** than ham (138.9 vs 71.0 chars)
- Spam messages use **23.9 words** on average vs 14.2 for ham
- **82.3%** of spam messages contain promotional keywords (free, win, prize, cash)
- Only **0.4%** of ham vs **13.5%** of spam contain URLs
- Spam frequently contains currency symbols (£, $, €) and large numbers

---

## Pipeline

```
Raw SMS Input
     ↓
Text Preprocessing
  • Lowercasing
  • URL → ' url ' token
  • Digits → ' num ' token
  • Remove punctuation & whitespace
     ↓
TF-IDF Vectorization
  • max_features = 8,000
  • ngram_range = (1, 2)   [unigrams + bigrams]
  • sublinear_tf = True
     ↓
Model Training (3 classifiers)
  • Naïve Bayes     MultinomialNB(alpha=0.1)
  • Logistic Reg.   LogisticRegression(C=5, lbfgs)
  • SVM             LinearSVC(C=1.0)
     ↓
5-Fold Stratified Cross-Validation
     ↓
Evaluation & Best Model Selection
     ↓
Live Inference
```

---

## Model Results

| Model               | Accuracy   | Precision  | Recall     | F1-Score   | CV F1 (±std)      |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ----------------- |
| Naïve Bayes         | 98.57%     | 95.40%     | 93.29%     | 94.44%     | 95.94% ±1.05%     |
| Logistic Regression | 98.57%     | 97.10%     | 92.28%     | 94.63%     | 94.91% ±1.15%     |
| **SVM (Best)**      | **98.92%** | **97.96%** | **94.00%** | **95.97%** | **95.54% ±0.79%** |

### Why SVM Wins

- Highest accuracy AND F1 across all metrics
- Lowest cross-validation standard deviation (most stable)
- LinearSVC maximizes the margin between spam and ham in the 8,000-dimensional TF-IDF space
- Handles the class imbalance (86.6% ham vs 13.4% spam) more robustly than probabilistic models

---

## Key Features

- **Full EDA** with 6 professional visualizations (dark purple theme)
- **Three fully tuned ML pipelines** (NB, LR, SVM) with scikit-learn
- **TF-IDF with bigrams** for richer text representation
- **5-fold stratified cross-validation** for robust evaluation
- **ROC curves with AUC** for all models
- **Live inference demo** with real sample messages
- **Interactive Web UI** with animated particle background
- **Pre-loaded notebook outputs** — no re-run required

---

## How to Run

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn notebook
```

### Launch Notebook

```bash
jupyter notebook notebook/Spam_SMS_Detection.ipynb
```

### Open UI

```bash
# Simply open in any browser:
open ui/index.html
```

### Quick Inference

```python
import pandas as pd, re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load data & train
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1','v2']]
df.columns = ['label','message']
df['label_enc'] = (df['label']=='spam').astype(int)

def preprocess(t):
    t=t.lower(); t=re.sub(r'http\S+|www\S+',' url ',t)
    t=re.sub(r'\d+',' num ',t); t=re.sub(r'[^a-z\s]',' ',t)
    return re.sub(r'\s+',' ',t).strip()

df['clean']=df['message'].apply(preprocess)
pipe = Pipeline([('tfidf',TfidfVectorizer(max_features=8000,ngram_range=(1,2),sublinear_tf=True)),
                 ('clf',LinearSVC(C=1.0))])
pipe.fit(df['clean'], df['label_enc'])

# Predict
msg = "Congratulations! You've won £1000! Call now to claim."
pred = pipe.predict([preprocess(msg)])[0]
print("SPAM" if pred==1 else "HAM")
```

---

## Technologies Used

| Category      | Tools                                        |
| ------------- | -------------------------------------------- |
| Language      | Python 3.11                                  |
| ML Framework  | scikit-learn 1.4+                            |
| NLP           | TF-IDF Vectorizer, CountVectorizer           |
| Models        | MultinomialNB, LogisticRegression, LinearSVC |
| Data          | pandas, NumPy                                |
| Visualization | Matplotlib, Seaborn                          |
| Notebook      | Jupyter Notebook                             |
| Frontend      | HTML5, CSS3, Vanilla JS (Canvas API)         |

---

## Author

**Karthikeyan Thirunavukkarasu**
Data Analytics Intern @ Cognifyz Technologies
Tiruchirappalli, Tamil Nadu, India

- LinkedIn: [linkedin.com/in/karthikeyan-thirunavukkarasu-2a2949305](https://linkedin.com/in/karthikeyan-thirunavukkarasu-2a2949305)
- GitHub: [github.com/karthiikofcl07](https://github.com/karthiikofcl07)

---

_Built with precision. Delivered with professionalism._

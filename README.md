
# Product Complaint Classification with Bi-LSTM
This project develops an automated system for classifying and routing financial complaints by analyzing their text content using natural language processing and machine learning methods.



## 📌 Project Overview

* Dataset: **162,421 complaints** across **5 financial products**.
* Primary Model: **Bidirectional LSTM (Bi-LSTM)** for text classification.
* Benchmarks: Bag-of-Words, n-grams, TF-IDF + Naive Bayes.
* Imbalance Handling: Random undersampling to improve fairness and accuracy.
* Full **text preprocessing pipeline** for cleaning real-world complaint data.

---

## Workflow

1. **Introduction** – Problem statement, dataset, objectives.
2. **Imports** – Core libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow/keras`, `nltk`.
3. **Data Loading & Analysis** – Reads `complaintsprocessed.csv`, inspects class distribution.
4. **Text Preprocessing** – HTML cleaning, stopword removal, lemmatization, regex-based text normalization.
5. **Imbalance Handling** – Undersampling for balanced training sets.
6. **Feature Engineering** – Bag-of-Words, n-grams, TF-IDF vectorization.
7. **Classical Models** – Naive Bayes classification with evaluation metrics.
8. **Deep Learning (Bi-LSTM)** –

   * Tokenization & padding
   * Bidirectional LSTM with dropout & batch normalization
   * Training & evaluation with precision, recall, F1-score
9. **Results & Visualizations** – Confusion matrices, class reports, accuracy/F1 comparisons.
10. **Model Saving** – Persists trained models and tokenizers with Pickle and Keras utilities.

---

## ✨ Key Features

* End-to-end **NLP pipeline**: ingestion → preprocessing → modeling → evaluation.
* **Robust preprocessing** for contractions, HTML tags, special symbols, casing, and stopwords.
* **Multiple baselines** to benchmark deep learning vs. classical NLP models.
* Comprehensive **metrics reporting**: accuracy, precision, recall, F1, per-class performance.

---

## ⚙️ Requirements

* Python 3.7+
* pandas, numpy, matplotlib, seaborn
* scikit-learn, nltk, regex, beautifulsoup4
* tensorflow, keras
* pickle (standard library)

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk tensorflow keras beautifulsoup4
```

---

## 🚀 Usage

1. Place the dataset file `complaintsprocessed.csv` in the project folder.
2. Launch the notebook:

   ```bash
   jupyter notebook
   ```
3. Open and run `main.ipynb` step by step.
4. Adjust preprocessing options or model hyperparameters as needed.
5. Outputs include:

   * Evaluation metrics and reports
   * Trained models & tokenizers
   * Class imbalance visualizations & confusion matrices

---

## 📊 Results

* **Bi-LSTM** achieves higher **accuracy and F1-scores** compared to TF-IDF and Bag-of-Words models.
* **Class imbalance** handled with undersampling leads to fairer predictions across categories.
* Notebook provides **transparent metrics and plots** for model comparison.



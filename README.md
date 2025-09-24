# CFCMS - Content Filtering and Classification Management System

## Overview
This repository contains the implementation of **CFCMS**, a machine learning-based text classification system.  
The project focuses on preprocessing textual data, extracting features using **TF-IDF** and **word embeddings**, 
and classifying content using machine learning models such as **Logistic Regression**.  

The system is designed to filter and classify text efficiently, making it useful for applications such as:  
- Fake news detection  
- Spam filtering  
- Sentiment or topic classification  

## Features
- Text preprocessing (tokenization, stemming, stopword removal, HTML parsing with BeautifulSoup).  
- Feature extraction using:
  - Bag-of-Words  
  - TF-IDF  
  - Word embeddings (Gensim Word2Vec / Keyed Vectors)  
- Classification with Scikit-learn models.  
- Evaluation with metrics such as accuracy, confusion matrix, ROC-AUC, and classification reports.  
- Parallelized execution using Python's `concurrent.futures`.  

## Project Structure
- `CFCMS.ipynb` : Main Jupyter Notebook containing the full implementation and experiments.  
- `requirements.txt` : Python dependencies required to run the project.  
- `.gitignore` : Git ignore file for clean repository management.  

## Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/CFCMS.git
cd CFCMS
pip install -r requirements.txt
```

## Usage
Open the Jupyter Notebook and run all cells:

```bash
jupyter notebook CFCMS.ipynb
```

Alternatively, convert the notebook into a Python script:

```bash
jupyter nbconvert --to script CFCMS.ipynb
python CFCMS.py
```

## Requirements
See [`requirements.txt`](requirements.txt) for full details. Key packages include:
- nltk
- gensim
- beautifulsoup4
- scikit-learn
- scipy

## Results
The notebook provides model evaluation with:  
- Accuracy scores  
- Confusion matrix  
- Classification report  
- ROC and AUC curves  

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.

## Sentiment Analysis towards COVID-19 on Twitter

This repository contains the code and report for a deep learning–based sentiment analysis project on COVID-19-related tweets.  
The work was completed as part of **COMP9444 Neural Networks & Deep Learning** (Group Project – Deep Learners).

The core analysis lives in the notebook `covid_19_sentiment_analysis.ipynb`, which compares:

- **Rule-based sentiment analysis** (e.g. VADER, TextBlob)
- **Classical machine learning models** (e.g. Naive Bayes, Logistic Regression, SVM, Random Forest)
- **Deep learning models** with different text representations (frequency-based vs neural-network-based embeddings such as Word2Vec / GloVe)

The goal is to explore **how deep learning and neural networks can improve sentiment analysis of COVID-19 tweets**.

---

## Team

| Name            | ID       | Email                               |
|-----------------|----------|-------------------------------------|
| Chenglong Wei   | z5375926 | chenglong.wei@student.unsw.edu.au   |
| Ziyi Ding       | z5610550 | z5610550@ad.unsw.edu.au             |
| Sidhanth Kafley | z5504979 | z5504979@ad.unsw.edu.au             |
| Yewei Huang     | z5459400 | z5459400@ad.unsw.edu.au             |
| Gorjan Muratov  | z5677486 | z5677486@ad.unsw.edu.au             |

---

## Repository Structure

- `covid_19_sentiment_analysis.ipynb` – main Jupyter/Colab notebook with the full pipeline (pre-processing, modelling, evaluation).
- `Project_Report.pdf` – complete written report describing the problem, methodology, experiments, and results.
- `Presentation_slides.pptx` – slide deck summarising the project.
- `README.md` – this file.

---

## Dataset

The notebook downloads and uses the **COVIDSenti** dataset:

- Source: `https://github.com/usmaann/COVIDSenti`
- File used: `COVIDSenti.csv`
- Labels: `neg` (negative), `neu` (neutral), `pos` (positive) – mapped to integers 0, 1, 2 in the code.
- Each row contains a tweet and its sentiment label.

The dataset is cloned inside the notebook via:

- `git clone https://github.com/usmaann/COVIDSenti.git`
- Then `COVIDSenti.csv` is loaded with `pandas`.

If you want to run locally, make sure `COVIDSenti.csv` is available under the expected path or adjust the path in the notebook.

---

## Methods Overview

The notebook explores two main dimensions:

- **1. Modelling approach**
  - Rule-based models (e.g. VADER, TextBlob)
  - Machine learning models (Naive Bayes, Logistic Regression, Linear SVM, Random Forest, etc.)
  - Deep learning architectures built with **PyTorch**.

- **2. Embedding / representation**
  - **Frequency-based**: bag-of-words and TF–IDF features (via `CountVectorizer` and `TfidfVectorizer` from `scikit-learn`).
  - **Neural-network-based**:  
    - Word2Vec (`gensim`)  
    - GloVe Twitter embeddings (downloaded from Stanford NLP and converted to Word2Vec format using `glove2word2vec`).

Tweets undergo **minimal pre-processing** to preserve sentiment-rich features:

- Convert text to lowercase
- Remove URLs
- Map string labels to integers
- Keep emojis, hashtags, and other special characters (these often carry sentiment)

The dataset is split into train, validation, and test sets (typical split used in the notebook: 70% / 20% / 10%).

---

## Environment and Dependencies

The notebook is designed to run in **Google Colab** and installs its own dependencies. Key Python libraries include:

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `nltk` (including VADER lexicon)
- `textblob`
- `torch`, `torchvision`, `tqdm`
- `gensim`
- `vaderSentiment`

In Colab, the following commands (already present in the notebook) are used:

- Download GloVe Twitter embeddings:
  - `wget http://nlp.stanford.edu/data/glove.twitter.27B.zip`
  - `unzip glove.twitter.27B.zip`
- Clone COVIDSenti:
  - `git clone https://github.com/usmaann/COVIDSenti.git`
- Install packages:
  - `pip install vaderSentiment`
  - `pip install gensim`

If you run the notebook locally, you should install the required packages in a virtual environment, for example:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt  # if you create one from the imports
```

Then open the notebook with Jupyter or VS Code and run the cells in order.

---

## How to Run the Notebook

### Option 1: Google Colab (recommended)

1. Open `covid_19_sentiment_analysis.ipynb` in Google Colab.  
   There is also an “Open in Colab” badge at the top of the notebook that links to the hosted version.
2. Ensure the runtime has access to the internet (for downloading datasets and embeddings).
3. Run all cells from top to bottom:
   - Setup and installations
   - Data loading and pre-processing
   - Model training and evaluation blocks
4. Inspect the printed metrics, classification reports, confusion matrices, and plots.

### Option 2: Local Jupyter environment

1. Clone this repository:

   ```bash
   git clone https://github.com/Sidhanth-Kafley/Sentiment-Analysis-towards-COVID-19-on-Twitter.git
   cd Sentiment-Analysis-towards-COVID-19-on-Twitter
   ```

2. (Optional but recommended) Create and activate a virtual environment.
3. Install Python dependencies (based on the imports listed in the notebook).
4. Make sure you have:
   - The COVIDSenti dataset (either cloned via `git clone https://github.com/usmaann/COVIDSenti.git` or downloaded manually).
   - The GloVe Twitter embeddings if you want to run the corresponding sections.
5. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

6. Open `covid_19_sentiment_analysis.ipynb` and execute the cells sequentially.

---

## Results and Interpretation

Detailed quantitative results (accuracy, precision, recall, F1 scores) and qualitative analysis are provided in:

- `covid_19_sentiment_analysis.ipynb` – includes tables, plots, and confusion matrices.
- `Project_Report.pdf` – discusses findings, comparisons between rule-based, ML, and deep-learning approaches, and the impact of different embeddings.
- `Presentation_slides.pptx` – high-level summary suitable for presentations.

At a high level, the experiments show how **deep learning models combined with learned embeddings** can outperform simpler baselines on the COVIDSenti tweet classification task, while also highlighting trade-offs in complexity, training time, and interpretability.

---

## Acknowledgements

- **COVIDSenti dataset** by Usmaan Ali and collaborators (`COVIDSenti` GitHub repository).
- **GloVe Twitter embeddings** by Stanford NLP Group.
- Various open-source libraries used in this project (`PyTorch`, `scikit-learn`, `nltk`, `gensim`, `vaderSentiment`, `TextBlob`, etc.).

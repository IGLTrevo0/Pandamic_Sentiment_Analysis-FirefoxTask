# COVID-19 Public Sentiment Analysis

## Project Overview
This project analyzes public sentiment expressed in pandemic-related tweets to track social trends and concern levels. It classifies tweets into three categories — Positive, Negative, and Neutral — using Natural Language Processing and Machine Learning techniques.

---

## Problem Statement
During the COVID-19 pandemic, millions of people expressed their opinions, fears, and hopes on social media. This project builds a system that automatically reads these tweets and categorizes the public sentiment to help understand what people were feeling and talking about.

---

## Dataset
- **Source:** COVID-19 Twitter Dataset
- **Total Tweets:** 119,800
- **Columns:** tweet text, sentiment label, compound score, positive score, negative score, neutral score, retweet count, favorite count
- **Target:** Sentiment column with 3 classes — pos, neg, neu
- **Note:** Sentiment scores were pre-calculated using VADER (Valence Aware Dictionary and Sentiment Reasoner)

---

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Model:** Naive Bayes Classifier
- **Text Processing:** TF-IDF Vectorizer

---

## Project Workflow

### Step 1 — Text Preparation
- Removed links, @mentions, hashtags, RT tags, numbers and special characters from raw tweets
- Converted all text to lowercase
- Created clean_text column for model input

### Step 2 — Exploratory Analysis
- Visualized sentiment distribution using bar chart and pie chart
- Identified most common words in each sentiment group using word frequency analysis

### Step 3 — Feature Extraction
- Used TF-IDF (Term Frequency Inverse Document Frequency) to convert cleaned tweet text into 500 numerical word features
- Each word becomes its own column with a score showing how important that word is in each tweet

### Step 4 — Model Training
- Split data into 80% training and 20% testing using stratified split
- Trained Naive Bayes Classifier which is specifically designed for text classification tasks

### Step 5 — Evaluation
- Measured accuracy score
- Generated classification report showing precision, recall and F1 score per class
- Plotted confusion matrix to visualize where the model made mistakes

---

## Why These Choices?

**Why Classification and not Regression?**
The target variable is a category (pos/neg/neu) not a number. Regression predicts numbers. Classification predicts categories. So classification is the correct approach here.

**Why Naive Bayes and not Logistic Regression?**
Logistic Regression gave only 47% accuracy. Naive Bayes is specifically built for text data and works perfectly with TF-IDF features giving much better accuracy in seconds.

**Why TF-IDF and not raw scores?**
Using pre-calculated VADER scores as features gave poor accuracy because the model was not actually reading the tweet content. TF-IDF lets the model learn directly from the words people used in their tweets.

**Why stratify=y in train test split?**
To make sure positive, negative and neutral tweets are equally represented in both training and testing sets so the model is not biased towards any one class.

---

## Results
- Model used: Naive Bayes Classifier
- Features: 500 TF-IDF word features
- Training samples: ~95,840
- Testing samples: ~23,960
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score

---

## Key Findings
- Neutral tweets were the most common in the dataset
- Negative tweets frequently mentioned words like deaths, cases, fear, india
- Positive tweets frequently mentioned words like vaccine, hope, help, support
- Neutral tweets were mostly informational containing words like cases, update, health

---

## How to Run
1. Install required libraries
```
pip install pandas numpy matplotlib seaborn scikit-learn
```
2. Place the dataset file in the same folder as the notebook
3. Open Covid_Sentiment_Analysis.ipynb
4. Run all cells from top to bottom in order

---

## Project Structure
```
├── Covid_Sentiment_Analysis.ipynb   # Main notebook
├── nlptask (1).csv                  # Dataset
└── README.md                        # Project documentation
```

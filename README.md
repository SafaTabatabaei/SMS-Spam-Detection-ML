# SMS-Spam-Detection-ML
Developed an efficient ML model to accurately classify SMS messages as "SPAM" or "HAM" (not spam). Leveraging NLP techniques for text processing and feature engineering 
# 📧 SMS Spam Detection using Machine Learning (NLP)

## Project Overview

This project focuses on building a robust Machine Learning model to classify SMS messages as either "Spam" or "Ham" (legitimate message). It demonstrates an end-to-end Natural Language Processing (NLP) pipeline, from raw text data to a deployed classification model.

## ✨ Key Features & Technologies

-   *Natural Language Processing (NLP):* Comprehensive text preprocessing including cleaning, tokenization, stop-word removal, and stemming/lemmatization (if applied).
-   *Text Vectorization:* Transforming text into numerical features using techniques like *TF-IDF (Term Frequency-Inverse Document Frequency)* or *CountVectorizer*.
-   *Machine Learning Models:* Implementation and comparison of various classification algorithms, such as:
    -   *Naive Bayes (MultinomialNB):* Often a strong baseline for text classification due to its simplicity and effectiveness.
    -   *Support Vector Machines (SVC):* Powerful algorithm for high-dimensional data.
    -   *Logistic Regression / RandomForest Classifier (Optional, if explored)*
-   *Model Evaluation:* Rigorous assessment of model performance using key metrics like:
    -   Accuracy
    -   Precision, Recall, F1-Score
    -   Confusion Matrix
-   *Data Visualization:* Illustrating insights from data and model performance.

*Technologies Used:*
Python | Pandas | Numpy | Scikit-learn | NLTK | Matplotlib | Seaborn

## 📊 Dataset

This project utilizes the *SMS Spam Collection Dataset*, which comprises over 5,000 legitimate (ham) and spam SMS messages.

-   *Source:* [https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
-   *File:* spam.csv (included in this repository)

## 🚀 How to Run the Project

1.  *Clone the repository:*
    bash
    git clone [https://github.com/YourUsername/SMS-Spam-Detection-ML.git](https://github.com/YourUsername/SMS-Spam-Detection-ML.git)
    cd SMS-Spam-Detection-ML
    
    (Replace YourUsername with your actual GitHub username)

2.  *Install dependencies:*
    bash
    pip install pandas numpy scikit-learn matplotlib seaborn nltk
    
    (Note: NLTK will require additional data downloads as prompted by the notebook.)

3.  *Run the Jupyter Notebook:*
    bash
    jupyter notebook spam_detection.ipynb
    
    (Replace spam_detection.ipynb with your notebook's actual filename)

## ✅ Results & Performance

(IMPORTANT:* Replace the placeholders below with your actual project results and add your screenshots!)*

-   *Model Accuracy:* Achieved an accuracy of *~98%* in classifying SMS messages.
-   *Key Metrics (Example for Naive Bayes):*
    -   Precision (Spam): ~96%
    -   Recall (Spam): ~88%
    -   F1-Score (Spam): ~92%


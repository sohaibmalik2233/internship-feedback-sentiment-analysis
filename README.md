# Internship Feedback Sentiment Analysis: Advanced NLP Classification

An end-to-end Natural Language Processing (NLP) machine learning pipeline built to automatically classify unstructured workplace and internship feedback into binary sentiment categories. This project handles raw text parsing, transforms linguistic parameters into numerical vectors, and utilizes an optimized classification model to achieve competitive accuracy on unseen validation text.

## 🏆 Performance Milestone
* **Algorithm:** Logistic Regression (via TF-IDF Vectorization)
* **Testing Accuracy:** 90.90%
* **Precision / Recall:** 0.91 / 0.91
* **F1-Score:** 0.91

---

## 🧠 Project Overview
Extracting actionable insights from hundreds of thousands of employee reviews requires capturing complex linguistic patterns across unstructured text. This repository outlines a reproducible workflow starting from raw, unverified review matrices to a fully functional predictive classification pipeline.

### Key Pipeline Stages:
1. **Data Preprocessing & Cleaning:** Managed structural missingness by dynamically isolating null text features, followed by strict structural labeling mapping "Pros" to a Positive class vector (1) and "Cons" to a Negative class vector (0). The master dataset was subsequently unified and randomized to prevent chronological training bias.
2. **Feature Engineering:** Implemented strict TF-IDF (Term Frequency-Inverse Document Frequency) transformations to map textual parameters into highly structured sparse numerical matrices. Confined the feature space to the top 5,000 statistical terms while aggressively filtering standard English stop-words to reduce dimensional noise.
3. **Model Selection:** Evaluated baseline estimators and ultimately selected an optimized **Logistic Regression** classifier to robustly calculate probabilistic thresholds and combat variance without overfitting the textual training data.

---

## 🛠 Tech Stack & Dependencies
* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Machine Learning & NLP Framework:** Scikit-Learn
* **Data Visualization:** Seaborn, Matplotlib

---

## 💻 Getting Started

### 1. Clone the Repository
    ```bash
    git clone [https://github.com/sohaibmalik2233/internship-feedback-sentiment-analysis.git](https://github.com/sohaibmalik2233/internship-feedback-sentiment-analysis.git)
    cd internship-feedback-sentiment-analysis
    ```

### 2. Environment Setup
Install the required dependencies via the provided text file to ensure version consistency:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Data Integration
Place the raw `glassdoor_reviews.csv` matrix into the `/data` directory. 
*(Note: This file is restricted from version control via `.gitignore` due to large file size constraints).*

### 4. Pipeline Execution
Navigate to the `/notebook` directory and execute the primary Jupyter Notebook (`01_feedback_sentiment_model.ipynb`) to initialize the data parsing, TF-IDF translation, and model training sequences.
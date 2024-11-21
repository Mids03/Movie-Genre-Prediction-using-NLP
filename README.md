# Movie Genre Classification

This project demonstrates a machine learning approach to classify movies into their respective genres based on their description.

## Dataset

The dataset used for this project can be accessed on Kaggle:  
[Movie Genre Classification Dataset](https://www.kaggle.com/code/imgowthamg/movie-genre-classification)

The dataset contains the following files:
- `train_data.txt`: Training data with movie titles, descriptions, and genres.
- `test_data.txt`: Test data with movie titles and descriptions.
- `test_data_solution.txt`: Solution file for the test data with genres included.

## Overview

The goal is to build a classification model that predicts the genre(s) of a movie based on the textual data. The pipeline involves data preprocessing, exploratory data analysis (EDA), feature extraction, model training, and evaluation.

## Methodology

1. **Data Loading**:
   - Read training and test datasets.
   - Perform initial exploration to understand the structure and distribution of the data.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize the genre distribution.
   - Explore relationships between titles, descriptions, and genres.

3. **Data Preprocessing**:
   - Clean and tokenize text data.
   - Convert text into numerical features using techniques like TF-IDF or word embeddings.

4. **Model Training**:
   - Use machine learning models (e.g., logistic regression, SVM, or neural networks) to predict genres.
   - Evaluate models using appropriate metrics such as accuracy, F1-score, and precision/recall.

5. **Evaluation**:
   - Test the model on unseen data.
   - Analyze the results and potential areas for improvement.

## Prerequisites

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
## Results
- The classification model achieves significant accuracy in predicting genres.
- Results are analyzed through visualizations and evaluation metrics.
## Notes
- **Multi-Label Classification**: As a movie can belong to multiple genres, this is treated as a multi-label classification problem.
- **Imbalanced Dataset**: Some genres may have fewer samples than others, requiring careful handling through techniques like oversampling or weighted loss functions.
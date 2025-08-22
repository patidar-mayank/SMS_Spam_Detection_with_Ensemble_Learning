# SMS Spam Detection with Ensemble Learning

This project demonstrates a machine learning pipeline for detecting spam SMS messages using an ensemble learning model. The core of the project is implemented in the Jupyter Notebook `SMS_Spam_Detection_with_Ensemble_Learning.ipynb`.

## Overview

The objective is to classify SMS messages as "spam" or "ham" (not spam) using a combination of text preprocessing, feature engineering, and an ensemble of classical machine learning models. The ensemble approach leverages the strengths of multiple algorithms to improve classification accuracy.

## Dataset

- The dataset used is a publicly available SMS spam collection in CSV format (`spam.csv`).
- It contains two main columns after cleaning:
  - `label`: Indicates whether the message is spam (`1`) or ham (`0`).
  - `text`: The actual SMS message.

## Installation

To run this project, please ensure you have the following dependencies installed:

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

Additionally, you will need to download the NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
```

## Usage

1. **Clone the repository** and navigate to the project directory.
2. **Place the `spam.csv` dataset** in the appropriate location (or update the path in the notebook).
3. **Open the Jupyter Notebook**:

   ```bash
   jupyter notebook SMS_Spam_Detection_with_Ensemble_Learning.ipynb
   ```

4. **Run all cells** to execute the workflow and see the results.

## Project Workflow

1. **Import Libraries:** Load necessary Python libraries for data manipulation, visualization, and machine learning.
2. **Load Dataset:** Read the SMS dataset into a Pandas DataFrame.
3. **Data Cleaning:**
    - Remove unnecessary columns with many missing values.
    - Rename columns for clarity.
    - Check for and handle missing values.
4. **Text Preprocessing:**
    - Convert text to lowercase, remove special characters and stopwords.
    - Create a new column, `cleaned_text`, with the cleaned messages.
5. **Label Encoding:** Convert text labels to numerical values (ham=0, spam=1).
6. **Feature Engineering:**
    - Use `TfidfVectorizer` to convert text data into numerical features.
    - Limit the number of features to 3000 and remove English stopwords.
7. **Train-Test Split:** Split the dataset into training (80%) and test (20%) sets.
8. **Model Building:**
    - Define three base models: Multinomial Naive Bayes, Logistic Regression, and Support Vector Classifier.
    - Combine them using a soft voting ensemble (`VotingClassifier`).
    - Train the ensemble model on the training data.
9. **Model Evaluation:**
    - Predict on the test set.
    - Evaluate using accuracy, confusion matrix, and classification report.

## Model Architecture

- **Voting Ensemble Model**: Combines predictions from:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Support Vector Classifier (SVC)
- **Voting Type**: Soft voting (uses predicted probabilities to make the final decision).

## Results

The ensemble model achieves high accuracy on the test set:

- **Accuracy:** ~97.8%
- **Precision, Recall, F1-score:** (see the notebook output for details)
- The confusion matrix and classification report show strong performance in both classes, with especially high precision for the "ham" class.

## Acknowledgments

- The SMS Spam Collection Dataset is provided by [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).
- The ensemble learning approach builds on standard scikit-learn models.

---

**Note:** For further exploration, feel free to modify the notebook to try different models, feature engineering techniques, or preprocessing steps!

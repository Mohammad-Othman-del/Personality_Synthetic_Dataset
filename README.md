
---

# Personality Prediction Project: Understanding People with Machine Learning

## 1. Overview

This project demonstrates how machine learning can be used to understand and predict personality types based on behavioral traits. It involves building and comparing three classic machine learning algorithms implemented entirely from scratch:

* Decision Tree
* Random Forest
* AdaBoost

The primary objective is to predict a person’s personality type — whether they are an **Extrovert**, **Introvert**, or **Ambivert** — based on various input characteristics, and to determine which model performs best.

---

## 2. Dataset

The dataset used in this project is named `Data.csv`. Each record represents a person, described by several personality-related traits.

### Example Features

* **social_energy**: How much energy a person gains from social interactions.
* **alone_time_preference**: How much a person enjoys being alone.
* **talkativeness**: How talkative or expressive the person tends to be.
* *(and other similar traits)*

The target column is **`personality_type`**, which contains one of three possible labels:

* `Extrovert`
* `Introvert`
* `Ambivert`

---

## 3. Project Structure

```
.
├── Data.csv
├── personality_prediction.py
└── README.md
```

The main logic resides in the Python script `personality_prediction.py`, which:

1. Loads and prepares the dataset.
2. Trains Decision Tree, Random Forest, and AdaBoost models.
3. Evaluates performance using Accuracy and F1-score.
4. Compares the models to determine the best-performing one.
5. Prints results and insights.

---

## 4. How the Code Works

### Step 1: Data Preparation

* The dataset is read using pandas.
* The features (traits) are separated from the target (`personality_type`).
* The data is split into:

  * **60% Training Set** – used to train the models.
  * **30% Validation Set** – used to tune and compare models.
  * **10% Test Set** – used for final evaluation on unseen data.

### Step 2: Model Training

Three algorithms are implemented and trained:

#### Decision Tree

A simple, interpretable model that splits the data into smaller decision points using Gini impurity.

#### Random Forest

An ensemble of many Decision Trees. Each tree is trained on a random subset of data and features to reduce overfitting and improve accuracy.

#### AdaBoost

An ensemble method that combines multiple weak learners (small decision trees) sequentially. Each new learner focuses on the errors of the previous ones.

### Step 3: Model Evaluation

Each model is evaluated using:

* **Accuracy**: The percentage of correct predictions.
* **F1-score**: A balance between precision and recall.

A custom **classification report** is generated showing precision, recall, and F1-score for each class.

---

## 5. How to Run the Experiment

### Requirements

Make sure you have the following Python packages installed:

```
numpy
pandas
scikit-learn
```

### Running the Code

1. Place your dataset file `Data.csv` in the same folder as the script.
2. Run the Python file:

   ```bash
   python personality_prediction.py
   ```
3. The program will:

   * Load and summarize the dataset.
   * Train all three models.
   * Compare validation results.
   * Select and test the best model.
   * Print performance metrics and insights.

---

## 6. Example Output Summary

The output will display:

* Data distribution among personality types.
* Validation and test performance (Accuracy and F1-score).
* The best model based on validation results.
* A detailed classification report for the top model.

Example format:

```
Decision Tree - Accuracy: 0.74, F1: 0.72
Random Forest - Accuracy: 0.81, F1: 0.80
AdaBoost      - Accuracy: 0.79, F1: 0.77
Best model: Random Forest
```

---

## 7. Results and Insights

### Decision Tree

* Easy to interpret and visualize.
* Performs reasonably well but can overfit the training data.

### Random Forest

* Generally provides the best performance.
* More stable and robust than a single decision tree.
* Reduces overfitting by averaging multiple trees.

### AdaBoost

* Performs well, especially when data has difficult-to-classify samples.
* Builds an ensemble of weak learners that focus on previous mistakes.

Overall, the **Random Forest** and **AdaBoost** models outperform the single Decision Tree, confirming that ensemble methods are more effective for personality prediction tasks.

---

## 8. Key Takeaways

* Ensemble methods like Random Forest and AdaBoost significantly improve prediction accuracy.
* Building models from scratch enhances understanding of core machine learning algorithms.
* Data preparation and evaluation design play a critical role in fair model comparison.

---

## 9. Future Improvements

* Add cross-validation to further improve reliability.
* Include feature importance visualization.
* Experiment with additional ensemble methods like Gradient Boosting or XGBoost.
* Deploy the best model as a simple web or API application.

---

## 10. Author

Developed as part of a project exploring interpretable machine learning and human personality modeling.
Feel free to modify, extend, or integrate this code into your own experiments.

---


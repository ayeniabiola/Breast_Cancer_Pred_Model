# Breast_Cancer_Pred_Model
# Breast Cancer Classification

## Project Overview

This project uses machine learning to classify breast cancer cases as **Malignant (M)** or **Benign (B)** based on measurements from breast cancer diagnostic data.

The project focuses on data preprocessing, model training, and model evaluation using Python and Scikit Learn.

## Dataset

The dataset contains **569 observations** and **30 numerical features** related to characteristics of cell nuclei.

The target variable is:

* `diagnosis`

  * `M` = Malignant
  * `B` = Benign

The following columns were removed during preprocessing:

* `id`
* `Unnamed: 32`

```python
X = can.drop(columns=['id', 'diagnosis', 'Unnamed: 32'])
y = can['diagnosis']
```

## Machine Learning Models

Two classification algorithms were used:

1. **Logistic Regression**
2. **Decision Tree Classifier**

### Logistic Regression

Logistic Regression was used as a classification model to predict whether a tumor is malignant or benign.

### Decision Tree

A Decision Tree Classifier was used to learn patterns in the dataset and classify observations into malignant or benign categories.

## Data Preprocessing

The dataset was divided into training and testing sets using `train_test_split`.

Feature scaling was applied to the Logistic Regression model using `StandardScaler`.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

## Model Evaluation

The models were evaluated using classification metrics such as:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

These metrics were used to assess how well the models distinguish between malignant and benign cases.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit Learn
* Jupyter Notebook

## Key Learning

This project helped me understand how classification models work in a practical machine learning problem, including:

* Preparing real world data
* Handling unnecessary columns
* Splitting data into training and testing sets
* Feature scaling
* Training classification models
* Evaluating model performance

## Disclaimer

This project is for **educational and machine learning practice purposes only**. The results should not be used as a substitute for professional medical diagnosis.

## Author

**Abiola Ayeni**

Aspiring AI and Data Professional

GitHub: [ayeniabiola](https://github.com/ayeniabiola)

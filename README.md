# Census Income Prediction

## Overview

This project applies supervised machine learning to the **U.S. Census Income Dataset** to predict whether an individual's annual income is greater than $50,000.

The project follows the machine learning lifecycle from exploratory data analysis and preprocessing through model training, hyperparameter optimization, evaluation, and comparison.

Two classification approaches were developed:

* **Logistic Regression**
* **Feedforward Neural Network**

The project also considers ethical and fairness implications associated with using demographic and socioeconomic data for predictive modeling.

## Problem Statement

The goal is to predict the binary income class:

* `<=50K`
* `>50K`

This type of prediction can demonstrate how organizations might use machine learning to process large datasets and support faster, data-driven decision-making.

## Dataset

The dataset contains demographic and employment-related information from the 1994 U.S. Census.

Features include information related to characteristics such as:

* Age
* Education
* Workclass
* Occupation
* Hours worked per week
* Native country
* Other demographic and employment attributes

The target variable is:

```text
income_binary
```

## Data Preparation

Several preprocessing steps were performed before modeling.

### Missing Values

Numerical missing values were handled using median imputation:

* Age
* Hours per week

Categorical missing values were replaced using the mode:

* Workclass
* Occupation
* Native country

### Feature Removal

The following features were removed:

* `race`
* `sex_selfID`
* `fnlwgt`

Race and sex were removed because of fairness and ethical concerns associated with using demographic characteristics in income prediction. `fnlwgt` was removed because it represents a census sampling weight rather than a meaningful predictive characteristic for this modeling task.

### Encoding and Scaling

Categorical variables were converted into numerical representations using **one-hot encoding**.

Features were standardized using **StandardScaler** before model training.

## Exploratory Data Analysis

The analysis examined:

* Class distribution
* Missing values
* Feature data types
* Summary statistics
* Education and income relationships
* Age distributions across income classes

The exploratory analysis showed that the target classes were imbalanced and that variables such as education and age were associated with income outcomes.

## Machine Learning Models

### Logistic Regression

Logistic Regression was selected as the primary traditional machine learning model because this is a binary classification problem and the model provides interpretable coefficients.

Hyperparameters were optimized using **GridSearchCV** with 5-fold cross-validation.

The search evaluated:

```text
C: 0.01, 0.1, 1, 10
solver: liblinear, lbfgs
```

### Neural Network

A feedforward neural network was implemented using **TensorFlow/Keras**.

Architecture:

```text
Input Layer
    ↓
Dense Layer — 32 neurons, ReLU
    ↓
Dense Layer — 16 neurons, ReLU
    ↓
Output Layer — 1 neuron, Sigmoid
```

Training configuration included:

* SGD optimizer
* Learning rate: `0.01`
* Binary cross-entropy loss
* 20 epochs
* 20% validation split

## Results

| Model               |   Accuracy |   F1 Score |
| ------------------- | ---------: | ---------: |
| Logistic Regression | **85.35%** |     66.74% |
| Neural Network      | **85.20%** | **67.93%** |

The two models performed very similarly.

Logistic Regression produced slightly higher overall accuracy, while the Neural Network produced a slightly higher F1 score.

Because the performance difference was small, Logistic Regression was considered the stronger deployment candidate due to its **simplicity, interpretability, and lower complexity**.

## Ethical Considerations

Income prediction can reproduce historical and social inequalities present in the underlying data.

Even when sensitive attributes such as race and sex are removed, other variables may still act as proxies for protected characteristics.

Incorrect predictions could disproportionately affect disadvantaged or minority populations, particularly if such systems were used in high-impact areas such as:

* Lending
* Hiring
* Economic opportunity

For this reason, predictive models should support human decision-making rather than independently determine access to important opportunities.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow
* Keras
* Jupyter Notebook

## Key Takeaways

This project provided hands-on experience with:

* Defining a machine learning problem
* Exploratory data analysis
* Data preprocessing
* Feature engineering
* Binary classification
* Hyperparameter optimization
* Neural network development
* Model evaluation
* Model comparison
* Fairness and ethical considerations

## Future Improvements

Future work could explore:

* Additional feature engineering
* Class-weighted training
* Oversampling minority classes
* Alternative neural network architectures
* Dropout and regularization
* Adam optimization
* Additional evaluation metrics
* Fairness and subgroup performance analysis

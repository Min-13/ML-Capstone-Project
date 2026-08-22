# Census Income Prediction

## ✨ Project Highlights

* **85.35% test accuracy** with Logistic Regression
* **67.93% F1 score** with the Neural Network
* Compared a traditional machine learning model with a feedforward neural network
* Used **GridSearchCV with 5-fold cross-validation** for Logistic Regression hyperparameter selection
* Applied missing-value imputation, feature selection, one-hot encoding, and feature scaling
* Considered **fairness and ethical implications** when selecting model features
* Evaluated both models using **accuracy and F1 score**

## 📌 Project Overview

This project uses the **1994 U.S. Census Income dataset** to develop a binary classification model that predicts whether an individual's annual income is **greater than $50,000**.

The project follows the machine learning lifecycle, including problem definition, exploratory data analysis, data preparation, model development, evaluation, and comparison.

Two approaches were developed:

1. **Logistic Regression** as the traditional machine learning model
2. **Feedforward Neural Network** using TensorFlow/Keras

The objective was not only to achieve strong predictive performance, but also to evaluate model complexity, interpretability, and potential ethical concerns associated with income prediction.

## 🔎 Data Exploration

Exploratory data analysis identified several important characteristics of the dataset:

* The dataset contains both **numerical and categorical features**.
* Missing values were present in `age`, `workclass`, `occupation`, `hours-per-week`, and `native-country`.
* The income classes were **imbalanced**, with substantially more individuals earning `$50K or less` than those earning more than `$50K`.
* Education and age showed meaningful relationships with income.
* Individuals with higher education levels, particularly those with a **Bachelor's degree**, showed higher representation among individuals earning above $50K.

The dataset also raised ethical concerns. Features such as race, sex, and native country can reflect or act as proxies for protected characteristics. To reduce the potential for unfair predictions, `race` and `sex_selfID` were removed before modeling.

<img width="618" height="432" alt="image" src="https://github.com/user-attachments/assets/8cd074c4-5302-4f23-aa47-965bc5fc86aa" />


## 🧠 Model Development

### Logistic Regression

Logistic Regression was selected because the problem is a binary classification task and the model provides relatively interpretable coefficients.

Data preparation included:

* Median imputation for missing numerical values
* Mode imputation for missing categorical values
* Removal of `race`, `sex_selfID`, and `fnlwgt`
* One-hot encoding of categorical features
* Standardization using `StandardScaler`
* Stratified train/test split

`GridSearchCV` with 5-fold cross-validation was then used to identify an effective combination of Logistic Regression hyperparameters.

The model coefficients were also examined to understand which features were most strongly associated with the predicted outcome.

<img width="763" height="387" alt="image" src="https://github.com/user-attachments/assets/8ae69b95-ec9b-4109-9780-7d1fdb30a3c9" />


### Neural Network

A feedforward neural network was developed using TensorFlow/Keras to provide a more complex comparison with Logistic Regression.

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

Training configuration:

* **Optimizer:** SGD
* **Learning rate:** 0.01
* **Loss:** Binary Cross-Entropy
* **Epochs:** 20
* **Validation split:** 20%

Training and validation loss and accuracy were monitored throughout training to evaluate model learning and potential overfitting.

<img width="504" height="333" alt="image" src="https://github.com/user-attachments/assets/452c114a-3c99-4eac-8e4e-c94ad28644d4" />


## 📊 Results & Key Findings

Both models achieved very similar overall performance.

| Model               |   Accuracy |   F1 Score |
| ------------------- | ---------: | ---------: |
| Logistic Regression | **85.35%** |     66.74% |
| Neural Network      | **85.20%** | **67.93%** |

### Key Findings

**Logistic Regression achieved the highest accuracy.**
Its 85.35% accuracy was slightly higher than the Neural Network's 85.20%.

**The Neural Network achieved a slightly higher F1 score.**
Its F1 score of 67.93% was higher than Logistic Regression's 66.74%, suggesting a modest improvement in balancing precision and recall for the positive class.

**The performance gap was small.**
The Neural Network did not provide a large enough performance improvement to clearly justify its additional complexity for this dataset.

**Interpretability favored Logistic Regression.**
The model's coefficients provide a clearer explanation of how individual features influence predictions, making it easier to communicate results to stakeholders.

**Class imbalance remains important.**
Because the dataset contains substantially more observations in the `<=50K` class, accuracy alone does not fully describe model performance. The F1 score provides an additional perspective.

## 🧾 Model Summary

| Aspect           | Logistic Regression           | Neural Network                 |
| ---------------- | ----------------------------- | ------------------------------ |
| Model Type       | Linear classifier             | Feedforward neural network     |
| Accuracy         | **85.35%**                    | 85.20%                         |
| F1 Score         | 66.74%                        | **67.93%**                     |
| Interpretability | High                          | Lower                          |
| Complexity       | Lower                         | Higher                         |
| Training         | Faster                        | More computationally intensive |
| Best Advantage   | Simplicity & interpretability | Slightly higher F1 score       |

Overall, **Logistic Regression was the preferred model** for this project. Although the Neural Network produced a slightly higher F1 score, the improvement was small compared with the additional complexity involved in building, training, and interpreting the model.

## 🚀 Next Steps

Future improvements could focus on both predictive performance and responsible model development.

* Experiment with **class weights or oversampling** to address class imbalance.
* Perform additional **feature engineering** to capture relationships between demographic and employment variables.
* Test other Logistic Regression configurations and classification thresholds.
* Experiment with alternative Neural Network architectures.
* Evaluate optimizers such as **Adam** and introduce techniques such as **dropout** or regularization.
* Conduct more detailed **fairness and subgroup performance analysis**.
* Compare additional evaluation metrics, particularly precision, recall, and confusion matrices.
* Investigate whether model performance remains consistent across different demographic and socioeconomic groups.

The next stage would be to improve predictive performance while ensuring that any resulting model is **interpretable, fair, and appropriate for real-world decision support**.

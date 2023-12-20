# KNN
The code you provided implements a K Nearest Neighbors (KNN) classifier for a dataset related to personal loan modeling. 

---

**K Nearest Neighbors (KNN):**
KNN is a supervised machine learning algorithm used for classification and regression tasks. It classifies a data point based on the majority class of its k-nearest neighbors in the feature space. The choice of the parameter k (number of neighbors) is crucial and can impact the model's performance.

---

Below is a breakdown of the code along with an explanation of each section:

**Importing Libraries:**

* The code begins by importing necessary libraries such as pandas, matplotlib, seaborn, and scikit-learn modules for preprocessing, model building, and evaluation.

**Loading Dataset:**

* The dataset "Bank_Personal_Loan_Modelling.csv" is loaded into a Pandas DataFrame.

**Data Exploration:**

* The basic information about the dataset is displayed using the info() method.
* The first few rows of the dataset are displayed using the head() method.

**Feature Extraction:**

* The data is divided into features (X) and the target variable (y).
* Label encoding is applied to convert categorical variables into numerical form.

**Defining Features and Target:**

* A subset of features (X) and the target variable (y) are defined.

**Train-Test Split:**

* The dataset is split into training and testing sets using the train_test_split function from scikit-learn.

**Building the KNN Model:**

* A K Nearest Neighbors classifier is created using KNeighborsClassifier from scikit-learn with n_neighbors=5.
* The model is trained on the training set using the fit() method.

**Model Evaluation:**

* The accuracy of the model is evaluated on both the training and testing sets using the score method.
* Predictions are made on the testing set, and the accuracy, classification report, and confusion matrix are displayed.

**Results Display:**

* The code displays the predicted values, accuracy, classification report, and confusion matrix for evaluating the performance of the KNN model.

**Visualization:**

* The confusion matrix is visualized using a heatmap.

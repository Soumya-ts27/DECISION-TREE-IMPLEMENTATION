# DECISION-TREE-IMPLEMENTATION

**COMPANY**: CODETECH IT SOLUTIONS

**NAME**: TARIMELA SRINIVASA SOUMYA

**INTERN ID**:CT12WJVV

**DOMAIN**: MACHINE LEARNING

**BATCH DURATION**: JANUARY 5th,2025 to APRIL 5th,2025

**MENTOR NAME**: NEELA SANTHOSH

This Python code implements a Decision Tree Classifier using the Wine dataset from the `sklearn.datasets` module. It performs several tasks, including loading the dataset, splitting it into training and testing sets, training a decision tree model, making predictions, calculating accuracy, and visualizing the decision tree.

### Step 1: Importing Libraries

The code imports essential libraries:
- `matplotlib.pyplot` for visualizing the decision tree.
- `sklearn.datasets` for loading the Wine dataset.
- `sklearn.tree.DecisionTreeClassifier` to create and train a decision tree classifier.
- `sklearn.tree.plot_tree` to visualize the trained decision tree.
- `sklearn.model_selection.train_test_split` to split the dataset into training and testing sets.
- `sklearn.metrics.accuracy_score` to evaluate the model's accuracy.

---

### Step 2: Loading the Dataset

The Wine dataset is a commonly used dataset in machine learning. It contains chemical measurements of 178 wine samples from three different wine cultivars. The dataset has:
- **13 Features** representing chemical properties like alcohol content, malic acid, ash, magnesium, and others.
- **3 Classes (Targets)** representing the different types of wine.

The code uses `load_wine()` to load the dataset into `wine`. The features (`X`) and target labels (`y`) are extracted using `wine.data` and `wine.target`.

---

### Step 3: Splitting the Dataset

Using the `train_test_split()` function, the data is divided into training and testing sets. Here:
- `test_size=0.2` specifies that 20% of the data will be used for testing.
- `random_state=42` ensures reproducibility by fixing the random seed.

This split allows the model to learn from the training set and evaluate its performance on unseen data using the test set.

---

### Step 4: Creating and Training the Decision Tree Model

A Decision Tree Classifier is initialized using `DecisionTreeClassifier()`. The parameters include:
- `criterion='gini'` specifies the Gini impurity as the splitting criterion. Gini impurity measures the impurity of a node by calculating how often a randomly chosen element would be incorrectly labeled.
- `max_depth=3` limits the tree depth to prevent overfitting and improve interpretability.
- `random_state=42` ensures consistent results.

The classifier is trained using the `fit()` method with the training data (`X_train`, `y_train`).

---

### Step 5: Making Predictions

Once the model is trained, it uses the `predict()` method to make predictions on the test set (`X_test`). The predicted labels are stored in `y_pred`.

---

### Step 6: Evaluating the Model

The accuracy of the model is evaluated using `accuracy_score()`, which compares the predicted labels (`y_pred`) with the actual labels (`y_test`). Accuracy is calculated as:

\[
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
\]

The result is printed using a formatted string with two decimal places.

---

### Step 7: Visualizing the Decision Tree

The decision tree is visualized using `plot_tree()`, which generates a graphical representation of the tree structure. The visualization includes:
- **Filled Colors**: Representing the class distribution at each node.
- **Feature Names**: Displayed using `wine.feature_names` for clarity.
- **Class Names**: Indicating the wine types using `wine.target_names`.

The `plt.figure(figsize=(12, 8))` adjusts the figure size to ensure clear visualization.

---

### Conclusion

This code effectively demonstrates how to implement a decision tree classifier for a multi-class classification problem. The use of Gini impurity for splitting nodes, limiting the tree depth, and visualizing the tree enhances the interpretability of the model. Decision trees are especially useful for understanding how decisions are made, making them valuable for domains requiring explainability.

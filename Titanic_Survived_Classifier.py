# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Data Importing
train_df = pd.read_csv('/kaggle/input/titanic/test.csv')
test_df = pd.read_csv('/kaggle/input/titanic/train.csv')


# # Data Inspection Training dataset
print('Train dataset:')
print(train_df.head()) # Display the first 5 rows of the train dataset
print('_____________________________________________')
print(train_df.tail()) # Display the last 5 rows of the train dataset
print('_____________________________________________')
print('Shape of train dataset:', train_df.shape) # Check the shape of the dataset
print('_____________________________________________')
print('Columns in train dataset:', train_df.columns) # Check the columns in the dataset
print('_____________________________________________')
print('Info of train dataset:', train_df.info()) # Check the info of the dataset
print('_____________________________________________')
print(train_df.describe(include = 'all')) # Check the summary of the dataset
print('_____________________________________________')
print('Missing values in train dataset:', train_df.isnull().sum()) # Check the missing values in the dataset
print('_____________________________________________')

# Data Inspection Testing dataset
print('Test dataset:')
print(test_df.head()) # Display the first 5 rows of the test dataset
print('_____________________________________________')
print(test_df.tail()) # Display the last 5 rows of the test dataset
print('_____________________________________________')
print('Shape of test dataset:', test_df.shape) # Check the shape of the dataset
print('_____________________________________________')
print('Columns in test dataset:', test_df.columns) # Check the columns in the dataset
print('_____________________________________________')
print('Info of test dataset:', test_df.info()) # Check the info of the dataset
print('_____________________________________________')
print(test_df.describe(include = 'all')) # Check the summary of the dataset
print('_____________________________________________')
print('Missing values in test dataset:', test_df.isnull().sum()) # Check the missing values in the dataset
print('_____________________________________________')


# Exploratory Data Analysis  - EDA
print('Distribution of the target variable:') # Check the people who survived
print(train_df['Survived'].value_counts())
print('_____________________________________________')
print('Distribution of the target variable with respect to Pclass:') # Check the people who survived by Pclass
print(train_df.groupby('Pclass')['Survived'].value_counts())
print('_____________________________________________')
print('Distribution of the target variable with respect to Embarked:') # Check the people who survived by Embarked
print(train_df.groupby('Embarked')['Survived'].value_counts())
print('_____________________________________________')
print('Distribution of the target variable with respect to Sex:') # Check the people who survived by Sex
print(train_df.groupby('Sex')['Survived'].value_counts())
print('_____________________________________________')
print('Distribution of the target variable with respect to SibSp:') # Check the people who survived by SibSp
print(train_df.groupby('SibSp')['Survived'].value_counts())
print('_____________________________________________')
print('Distribution of the target variable with respect to Parch:') # Check the people who survived by Parch
print(train_df.groupby('Parch')['Survived'].value_counts())
print('_____________________________________________')
print('Distribution of the target variable with respect to Age:') # Check the people who survived by Age
print(train_df.groupby('Age')['Survived'].value_counts())
print('_____________________________________________')
print('Distribution of the target variable with respect to Fare:') # Check the people who survived by Fare
print(train_df.groupby('Fare')['Survived'].value_counts())
print('_____________________________________________')


# Exploratory Data Analysis Visualization - EDA Visualization
# Check the people who survived
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# Countplot for people who survived
sns.countplot(x='Survived', data=train_df, palette=['red', 'blue'], ax=axes[0])
axes[0].set_ylabel('People', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
axes[0].set_xlabel('Survived', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
axes[0].set_xticklabels(['Died', 'Survived'], fontsize=12, fontname='Times New Roman')
axes[0].set_title('Countplot for Survived People', fontsize=16, fontweight='bold', fontname='Times New Roman')
# Pie chart for Survived People
train_df['Survived'].value_counts().plot.pie(autopct='%1.1f%%', colors=['red', 'blue'], labels=['Died', 'Survived'], ax=axes[1])
axes[1].set_ylabel('')
axes[1].set_title('Pie chart for Survived People', fontsize=16, fontweight='bold', fontname='Times New Roman')
plt.tight_layout()
plt.show()

# Countplot for people who survived by Pclass
fig, axes = plt.subplots(1,figsize=(12, 6))
sns.countplot(x='Pclass', hue='Survived', data=train_df, palette=['red', 'blue'])
axes.set_ylabel('People', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
axes.set_xlabel('Pclass', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
axes.set_xticklabels(['1st class', '2nd class', '3rd class'], fontsize=12, fontname='Times New Roman')
axes.set_title('Countplot for Survived People by Pclass', fontsize=16, fontweight='bold', fontname='Times New Roman')
axes.legend(['Died', 'Survived'], title='People', loc='upper right')
plt.tight_layout()
plt.show()

# Countplot for people who survived by Embarked
fig, axes = plt.subplots(1, figsize=(12, 6))
sns.countplot(x='Embarked', hue='Survived', data=train_df, palette=['red', 'blue'])
axes.set_ylabel('People', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
axes.set_xlabel('Embarked', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
axes.set_xticklabels(['Cherbourg', 'Queenstown', 'Southampton'], fontsize=12, fontname='Times New Roman')
axes.set_title('Countplot for Survived People by Embarked', fontsize=16, fontweight='bold', fontname='Times New Roman')
axes.legend(['Died', 'Survived'], title='People', loc='upper right')
plt.tight_layout()
plt.show()

# Countplot for people who survived by Sex
fig, axes = plt.subplots(1, figsize=(12, 6))
sns.countplot(x='Sex', hue='Survived', data=train_df, palette=['red', 'blue'])
axes.set_ylabel('People', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
axes.set_xlabel('Sex', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
axes.set_xticklabels(['Male', 'Female'], fontsize=12, fontname='Times New Roman')
axes.set_title('Countplot for Survived People by Sex', fontsize=16, fontweight='bold', fontname='Times New Roman')
axes.legend(['Died', 'Survived'], title='People', loc='upper right')
plt.tight_layout()
plt.show()


# Data Preprocessing
# Fill in the missing values for the Training dataset
train_df['Age'] = train_df['Age'].interpolate(method='linear')
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df = train_df.dropna(subset=['Embarked'])
print('Missing values in train dataset:', train_df.isna().sum()) # Check the missing values in the Training dataset
print('_____________________________________________')

# Fill the missing values for Testist dataset
test_df['Age'] = test_df['Age'].interpolate(method='linear')
test_df['Fare'] = test_df['Fare'].interpolate(method='linear')
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
print('Missing values in test dataset:', test_df.isna().sum()) # Check the missing values in the Testing dataset
print('_____________________________________________')


# Correlation Visualization
correlation = train_df[['Survived', 'Sex', 'Pclass', 'Fare', 'Parch', 'Age', 'SibSp']].corr() # Compute the correlation matrix
# Create a figure heatmap for Correlation
fig, axes = plt.subplots(1, figsize=(16, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.3f', ax=axes)
axes.set_title('Correlation between the variables', fontsize=16, fontweight='bold', fontname='Times New Roman')
plt.tight_layout()
plt.show()

# Covariance Visualization
covariance = train_df[['Survived', 'Sex', 'Pclass', 'Fare', 'Parch', 'Age', 'SibSp']].cov() # Compute the covariance matrix
# Create a figure heatmap for Covariance
fig, axes = plt.subplots(1, figsize=(16, 6))
sns.heatmap(covariance, annot=True, cmap='coolwarm', fmt='.3f', ax=axes)
axes.set_title('Covariance between the variables', fontsize=16, fontweight='bold', fontname='Times New Roman')
plt.tight_layout()
plt.show()


# Create Machine Learning Model
features = ['Sex', 'Pclass', 'Fare', 'Parch',' Age', 'SibSp'] # Features for model
X = train_df[features]
y = train_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Split the data into training and testing sets

# Define the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier()
}
# Train and evaluate each model
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results[model_name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_prob)
    }
# Print the results
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print('_____________________________________________')
# Select the best model based on the evaluation metrics
best_model = RandomForestClassifier()  # Replace with your best model
best_model.fit(X_train, y_train)


# Testing Model
test_df['Survived'] = best_model.predict(test_df[features]) # Make predictions on the test dataset
submission = test_df[['PassengerId','Survived']] # Create a DataFrame for submission


# Create a figure barplot for Survived by Sex for Testing Dataset
fig, axes = plt.subplots(figsize=(8, 6))
sns.barplot(x='Sex', y='Survived', data=test_df, palette=['red', 'blue'])
axes.set_ylabel('Survival Probability', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
axes.set_xlabel('Sex', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
axes.set_xticks([0, 1])
axes.set_xticklabels(['Male', 'Female'], fontsize=12, fontname='Times New Roman')
axes.set_title('Classification Survival by Gender', fontsize=16, fontweight='bold', fontname='Times New Roman')
plt.tight_layout()
plt.show()


submission.to_csv('submission.csv', index=False) # Save the DataFrame to a CSV file
print("Submission file saved as 'submission.csv'")

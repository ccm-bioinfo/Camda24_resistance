# -*- coding: utf-8 -*-

"""
Created on Tue May 21 15:56:21 2024

@author: Ronald y Alejandro
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import confusion_matrix
matrix_file = "C:/Users/Alejandro/Desktop/Data/ResistanceCountStrict.tsv"
data = pd.read_csv(matrix_file, sep='\t', index_col=0)
X = data.iloc[:, 5:].values  
y = data['phenotype'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
numeric_features = list(range(5, X.shape[1]))  
numeric_transformer = StandardScaler()
non_numeric_features = [0, 1, 2, 4]  
non_numeric_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('non_num', non_numeric_transformer, non_numeric_features)
    ])
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier())])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

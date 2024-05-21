# -*- coding: utf-8 -*-

"""
Created on Tue May 21 15:56:21 2024

@author: Ronald y Alejandro
"""
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

"Clasificación Random Forest + PCA sobre archivo de PangenomeCountCiprofloxacin.tsv"

df = pl.read_csv("C:/Users/alejandro.sierra/Desktop/Data/PangenomeCountCiprofloxacin.tsv",separator="\t").to_pandas()
df = df[df['phenotype'].notna()] 
df = df.dropna()  

"codificación"
label_encoder = LabelEncoder()
df['accession'] = label_encoder.fit_transform(df['accession'])
df['genus'] = label_encoder.fit_transform(df['genus'])
df['species'] = label_encoder.fit_transform(df['species'])
df['phenotype'] = label_encoder.fit_transform(df['phenotype'])
feature_cols = df.columns.tolist()
feature_cols.remove('phenotype')
feature_cols.remove('measurement_value')
X = df[feature_cols]
y = df['phenotype']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

"reduccón de dimensionalidad y entrenamiento"
pca = PCA(n_components=150)
X_train_numeric = X_train[numeric_cols]
X_test_numeric = X_test[numeric_cols]
X_train_pca = pca.fit_transform(X_train_numeric)
X_test_pca = pca.transform(X_test_numeric)
X_train_pca_df = pd.DataFrame(X_train_pca, index=X_train.index, columns=[f'PCA_{i}' for i in range(X_train_pca.shape[1])])
X_test_pca_df = pd.DataFrame(X_test_pca, index=X_test.index, columns=[f'PCA_{i}' for i in range(X_test_pca.shape[1])])
X_train = X_train.drop(columns=numeric_cols).join(X_train_pca_df)
X_test = X_test.drop(columns=numeric_cols).join(X_test_pca_df)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

"Reporte"
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Classification Report:\n{report}')


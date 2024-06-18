

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:27:48 2024

@author: Alejandro, Ronald y Kotaro
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
def cond_mutual_info(x, y, z):
    return mutual_info_score(x, y) - mutual_info_score(x, z)

"Cargamos los datos y arreglamos la matriz para que la última columna sean 1 o -1"
data2 = pd.read_csv('C:/Users/Alejandro/Desktop/CAMDA2024/ResistanceJoinedStrictBiofiltered.tsv.gz', sep='\t',compression="gzip")
df = data2.drop(data2.columns[0], axis=1)
df = df.drop(columns=['genus','species','antibiotic','measurement_value'])
column = df.pop('phenotype')
df['phenotype'] = column
df = df.dropna()
df['phenotype'] = df['phenotype'].map({'Resistant': 1, 'Susceptible': -1})


"número de columnas"
c=348
"número de características seleccionadas"

Th = 3
"Tamaño de coalición"
p = 3
w = np.ones(c-2)
sum_RR = np.zeros(c-2)
lf = np.zeros(c-2)
flag = np.zeros(c-2)
"Indice de Banzhaf"
Banzhaf_power = np.zeros(c-2)
list_z = []
col_added = []
t = 1
CMI = 0
MI = 0
"Calcular los coeficientes de correlación y Tanimoto"
for i in range(1, c-1):
    summation = 0
    corre, _ = pearsonr(df.iloc[:, i], df.iloc[:, c-1])
    for j in range(1, c-1):
        if i != j:
            Tanimoto_coeff = pdist(np.array([df.iloc[:, i], df.iloc[:, j]]), 'jaccard')
            summation += Tanimoto_coeff
    Tanimoto_coeff_avg = summation/(c-3)
    sum_RR[i-1] = corre + Tanimoto_coeff_avg

while t <= Th:
    for i in range(c-2):
        if flag[i] != 1:
            lf[i] = sum_RR[i] * w[i]
    "seleccionar la característica con mayor lf"
    
    maximum = 0
    index = -1
    for i in range(c-2):
        if flag[i] != 1 and maximum < lf[i]:
            maximum = lf[i]
            index = i
    
    flag[index] = 1
    list_z.append(df.iloc[:, index+1])
    col_added.append(df.columns[index+1])
    
    len_col = len(col_added)
    "Calcular el índice de Banzhaf"
    
    if t != Th:
        for x in range(c-2):
            if flag[x] != 1:
                combin = 0
                for y in range(1, p+1):
                    if len_col >= y:
                        aa = [list(z) for z in itertools.combinations(col_added, y)]
                        combin += len(aa)
                        for g in range(len(aa)):
                            if df.columns[index+1] in aa[g]:
                                h = aa[g]
                                count = 0
                                sum_col_MI = 0
                                for q in range(y):
                                    col_posit = df.columns.get_loc(h[q])
                                    sum_col_MI += (cond_mutual_info(df.iloc[:, col_posit], df.iloc[:,c-1], df.iloc[:, x+1]) - 
                                                   mutual_info_score(df.iloc[:, col_posit], df.iloc[:,c-1]))
                                sum_col_MI /= y
                                for v in range(y):
                                    posit = df.columns.get_loc(h[v])
                                    CMI = cond_mutual_info(df.iloc[:, posit], df.iloc[:,c-1], df.iloc[:, x+1])
                                    MI = mutual_info_score(df.iloc[:, posit], df.iloc[:,c-1])
                                    if CMI > MI:
                                        count += 1
                                if sum_col_MI >= 0 and count >= np.ceil(y / 2):
                                    Banzhaf_power[x] += 1
                w[x] += Banzhaf_power[x] / combin
    t += 1
"características seleccionadas"

list_z.append(df.iloc[:,c-1])
selected_df = pd.concat(list_z, axis=1)


"Clasificación tomando las características/columnas seleccionadas"

data = selected_df
"características por considerar y target (última columna)"
X = data.iloc[:, :-1]  
y = data.iloc[:, -1]   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"Clasificador SVM con kernel Gaussiano"
svm_classifier = SVC(kernel='rbf')  
svm_classifier.fit(X_train, y_train)
"Evaluar el modelo"
y_pred = svm_classifier.predict(X_test)
"Calcular accuracy"
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

"Reporte de clasificación"
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)

"Matriz de confusión"
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
plt.figure(figsize=(10, 7))
class_names = ['Susceptible', 'Resistant']  # Adjust class names as needed
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('MC2OGT.png',format='png')
plt.show()

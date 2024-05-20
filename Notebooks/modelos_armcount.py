# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:51:27 2024

@author: Ronald and  Alejandro
"""
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# Ruta actual del directorio de trabajo
ruta_actual = os.getcwd()
print("Ruta actual:", ruta_actual)

# Cambiar la ruta a un directorio diferente a la que se necesita
nueva_ruta = 'C:/users/ronal/Documents/CIMAT/Segundo semestre/Microbioma/Conteo tabalas/AmrCount'
os.chdir(nueva_ruta)


# Cargo los datos de la tabla 

data2 = pd.read_csv('Amr_Count.tsv', sep='\t')

# Para ver el tipo de datos

# Todas las columnas a partir de "measurement_value" son flotantes
# su suma da justo 549
 
sum(data2.loc[:, "measurement_value":].dtypes == "float64")

plt.hist(data2["genus"])

#Ahora proponemos hacer un sdv truncado para reducir la dimensionalidad de los datos
# segun la ratio de varianza explicada





from sklearn.decomposition import TruncatedSVD, NMF


# Solo uso los datos de los conteos de genes para ver cuales son mas
# representativos


X = data2.loc[:,"3000620":]


# realiza factorizacion SVD o NMF (funcion del profe)
def get_factorization(data, n_comp=100, nmf=True):
    if nmf==False:
        fact_model = TruncatedSVD(n_components=n_comp)
        fact_model.fit(data)
    else:
        fact_model = NMF(n_components = n_comp, init=None, max_iter=12000)
        fact_model.fit(data)
    
    return fact_model






#Aplicamos SDVT
data_svd=get_factorization(data=X,n_comp=200, nmf=False)

plt.style.use('seaborn-whitegrid')

plt.plot(np.cumsum(data_svd.explained_variance_ratio_))
plt.scatter(100, sum(data_svd.explained_variance_ratio_[:100]), label='93%', color='red', marker='o')
plt.scatter(150, sum(data_svd.explained_variance_ratio_[:150]), label='94%', color='blue', marker='o')
plt.scatter(200, sum(data_svd.explained_variance_ratio_[:200]), label=' 96%', color='green', marker='o')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.legend(loc='lower right')



#Con lo cual concluimos quedarnos con n=150 componentes dado que explicamos alrededor
# del 94% de la varianza  explicada por las variables



fact_model = TruncatedSVD(n_components=150)
data_tsdv=fact_model.fit_transform(X)

#Ahora ya con matriz de covariables reducida en dimensionalidad
# procedemos a probar modelos para loc cual sepramos nuestos conjuntos 
# de prueba y entrenamiento como vemos ahora





df2 =  pd.DataFrame(data_tsdv)
df = data2.loc[:,:"measurement_value"]



# Nueva matrix unida ya bajada la dimensionalidad de sus variables

datos = pd.concat([df, df2], axis=1)



# Ahora separamos entre datos que si tiene las variables objetivo y los que no



# los datos que si tiene phenotyoe son 5761
# sum(datos["phenotype"].isin(["Susceptible","Resistant"]))

datos_train = datos[datos["phenotype"].isin(["Susceptible","Resistant"])]


# los que no tiene  phenotype son 760 
# sum(datos["phenotype"].isna())
datos_test = datos[datos["phenotype"].isna()]


##########################################################################
#    Analisis de datos Salmonella con el fin de obtener un resultado geenrla



datS= datos_train[datos_train.loc[:,"genus"]=="Salmonella"].loc[:,0:]

y = np.array(datos_train[datos_train.loc[:,"genus"]=="Salmonella"].loc[:,"phenotype"]).ravel()


from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
#Ahora procedemos a hacer el encondig de la variable objetivo




le = preprocessing.LabelEncoder()
le.fit(y)
#le.classes_
y_cat = le.transform(y)
#le.inverse_transform(y_cat[:5])



# Se usa el argumento stratify cuando tenemos calses desbalanceadas 
# se garantiza que la proporción de las clases en el conjunto de entrenamiento

X_train, X_test, y_train, y_test = train_test_split(datS, y_cat, test_size=0.2, stratify=y_cat, random_state=0)

from sklearn.linear_model import LogisticRegression


# Crea un modelo de regresión logística
model = LogisticRegression( max_iter=10000)

# Entrena el modelo utilizando los datos de entrenamiento
model.fit(X_train, y_train)


# Realiza predicciones en los datos de prueba
y_pred = model.predict(X_test)

# Calcula la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Muestra la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

from sklearn.metrics import classification_report




model = SVC(class_weight='balanced', random_state=42,C=0.3)
model.fit(X_train, y_train)

# Realiza predicciones en los datos de prueba
y_pred = model.predict(X_test)

# Calcula la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Muestra la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)



print(classification_report(y_test, y_pred))




#Agrramos los datos de prueba y hacemos la prediccion


datSt = datos_test[datos_test.loc[:,"genus"]=="Salmonella"].loc[:,0:]


yt = np.array(datos_test[datos_test.loc[:,"genus"]=="Salmonella"].loc[:,"phenotype"]).ravel()





# Realiza predicciones en los datos de prueba
y_pred = model.predict(datSt)

yprueba = le.inverse_transform(y_pred)


# Calcula la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)












###############################################################################
###############################################################################


##########################################################################
#    Analisis de datos con todos con el fin de obtener un resultado geenrla



dat = datos_train.loc[:,0:]

yf = np.array(datos_train.loc[:,"phenotype"]).ravel()


from sklearn import preprocessing

from sklearn.metrics import confusion_matrix
#Ahora procedemos a hacer el encondig de la variable objetivo




le = preprocessing.LabelEncoder()
le.fit(yf)
#le.classes_
yf_cat = le.transform(yf)
#le.inverse_transform(y_cat[:5])



# Se usa el argumento stratify cuando tenemos calses desbalanceadas 
# se garantiza que la proporción de las clases en el conjunto de entrenamiento

X_train, X_test, y_train, y_test = train_test_split(dat, yf_cat, test_size=0.2, stratify=yf_cat, random_state=0)


################### Usando SVC

model = SVC(class_weight='balanced', random_state=42,C=0.3)
model.fit(X_train, y_train)

# Realiza predicciones en los datos de prueba
y_pred = model.predict(X_test)

# Calcula la precisión del modelo
accuracysvm = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracysvm)

#   0.7363399826539462 valore de accuracy


# Muestra la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)



print(classification_report(y_test, y_pred))




#Agrramos los datos de prueba y hacemos la prediccion


datp = datos_test.loc[:,0:]


yfp = np.array(datos_test.loc[:,"phenotype"]).ravel()





# Realiza predicciones en los datos de prueba
y_pred = model.predict(datp)

ysvm = le.inverse_transform(y_pred)


# Calcula la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



#############################################################################
#############################################################################
####################### AHORA APLICAMOS OTRO MODELO LLAMDADO 
#############  randomforest el cual es mas robusto para datos desbalanceados



#############################################################################
# Ahora uso random forest para datos desbalanceados

from sklearn.metrics import classification_report

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Realiza predicciones en los datos de prueba
y_pred = model.predict(X_test)

# Calcula la precisión del modelo
accuracyRF = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracyRF)


# 0.8889852558542931 accuracyRF

# Muestra la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


print(classification_report(y_test, y_pred))


#Agrramos los datos de prueba y hacemos la prediccion


datp = datos_test.loc[:,0:]

yfp = np.array(datos_test.loc[:,"phenotype"]).ravel()



# Realiza predicciones en los datos de prueba
y_pred = model.predict(datp)

yRF = le.inverse_transform(y_pred)








#Ahora exportamos los resultados


final = datos_test.loc[:,:"antibiotic"]

final["phenotype"] = yprueba


final = final.rename(columns={'phenotype': 'phenotype(RForest)','antibiotic':'phenotype(SVM)'})

final["phenotype(SVM)"] = ysvm


    
# Exportar el DataFrame a un archivo CSV
final.to_csv('archivo.csv', index=False)



























































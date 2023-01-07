# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 18:24:10 2023

@author: FURKAN
"""

#*************************Kütüphaneler*********************************
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")

#*************************Veri Önişleme*********************************
# Datamızı import edelim ve gereksiz sütunları çıkaralım
data = pd.read_csv('data.csv')
df = pd.DataFrame(data)
df.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)

# Kategorik verileri logic değerlere dönüştürdük
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
from sklearn import preprocessing
df['diagnosis']= label_encoder.fit_transform(df['diagnosis'])

# Y verisine diagnosis sütununu atıp X'den çıkardık.
X = df.drop('diagnosis', axis = 1)
y = df["diagnosis"]

# Train ve Test verilerimizi böldük
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

# ******** Standart Sapmaya göre veriyi ölçeklendirdik********#
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)


#*****************************Fonksiyonlar************************************#
def simulatedAnnealing(X, y,model):
    # Bu fonksiyonda özellik seçimleri ile model doğruluk değerimizi arttırmayı hedefliyorum.
    T=1000
    alpha=0.99
    # Özelliklerin isimlerini aldık
    featureNames = list(X.columns)  
    # Bu kod satırı, "featureNames" isimli bir liste dizisinin içindeki 
    #özellik adlarını kullanarak bir sozluk oluşturur. Bu sözlükte,
    #her özellik adına bir anahtar eşleştirilmiş ve bu anahtarların değerleri 0 olarak ayarlanır. 
    firstX = {name: 0 for name in featureNames}  
    # En yüksek doğruluk değerine sahip x değerini sakladık
    bestX = firstX.copy()  
    # En yüksek doğruluk değerini sakladık
    bestAcc = 0  
    
    # Sıcaklığı azaltarak simulated annealing algoritmasını çalıştırdık
    for t in range(T):
        # Özellik seçim değişkenlerinin değerlerini rastgele değiştirdik
        for name in featureNames:
            firstX[name] = random.randint(0, 1)
            
        # Seçilen özellikleri belirledik ve modeli eğittik
        selectedFeatures = []
        for name in featureNames:
            if firstX[name] == 1:
                selectedFeatures.append(name)
                
        X_trainSelected = X_train[selectedFeatures]
        X_testSelected = X_test[selectedFeatures]
        model.fit(X_trainSelected, y_train)
        
        # Modelin doğruluk oranını hesapladık
        y_pred = model.predict(X_testSelected)
        acc = accuracy_score(y_test, y_pred)

        # Eğer modelin doğruluk oranı en yüksek doğruluk değerinden yüksekse, x değerini güncelledik
        if acc > bestAcc:
            bestX = firstX.copy()
            bestAcc = acc
        else:
            # Eğer modelin doğruluk oranı en yüksek doğruluk değernden düşükse ve
            # rastgele bir sayının değeri T/t değerinden küçükse, x değerini eski haline döndürdük
            p = np.exp((acc - bestAcc) / (T / t))
            # 0 ile 1 arasında, rastgele ondalıklı sayı ürettik.
            if random.uniform(0, 1) < p:
                firstX = bestX.copy()
                # Sıcaklığı azalttık
                T *= alpha
            # En yüksek doğruluk değerine sahip x değerinin seçilen özelliklerini döndürdük
            selectedFeatures = []
            for name in featureNames:
                if bestX[name] == 1:
                    selectedFeatures.append(name)
    confusionChart(y_test,y_pred)
    return selectedFeatures,bestX,bestAcc
#-----------------------------------------------------------------------------
def confusionChart(y_test,y_pred):
    import seaborn as sns
    confusionChart = confusion_matrix(y_test, y_pred)
    sns.heatmap(confusionChart, annot=True, fmt="d")
    plt.title("Confusion Chart")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
#-----------------------------------------------------------------------------
# En iyi k değerini bulmak için KNN modelimi birden fazla kez çalıştırıyorum.
def bestK(X_train, X_test, y_train, y_test):
    valueOfK = {}
    for i in range(15):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state = 42)
        tempKNN = KNeighborsClassifier(n_neighbors=i+1)
        tempKNN.fit(X_train, y_train)
        tempPred = tempKNN.predict(X_test)
        tempAcc = accuracy_score(y_test, tempPred)
        valueOfK[i] = tempAcc
    return valueOfK
    
from sklearn.metrics import accuracy_score, confusion_matrix

#*************************Optimizasyon Öncesi*********************************#
# KNN modelimizi uyguladık.
bestKValue = bestK(X_train, X_test, y_train, y_test)
print("KNN ->")
knnModel = KNeighborsClassifier(n_neighbors=9)
knnModel.fit(X_train, y_train)
y_pred = knnModel.predict(X_test)
preAcc = accuracy_score(y_test, y_pred)
print("KNN Optimizasyon Oncesi Accuracy ->",preAcc)
# # KNN Confusion Chart
confusionChart(y_test,y_pred)

# Logistic Regression modelimizi uyguladık.
print("Logistic Regression ->")
logRegModel = LogisticRegression()
logRegModel.fit(X_train, y_train)
y_pred = logRegModel.predict(X_test)
preAcc = accuracy_score(y_test, y_pred)
print("Logistic Regression Optimizasyon Oncesi Accuracy ->",preAcc)

# # Logistic Regression Confusion Chart
confusionChart(y_test,y_pred)

# Naive Bayes modelimizi uyguladık.
print("Naive Bayes ->")
navieBayesModel = GaussianNB()
navieBayesModel.fit(X_train, y_train)
y_pred = navieBayesModel.predict(X_test)
preAcc = accuracy_score(y_test, y_pred)
print("Naive Bayes Optimizasyon Oncesi Accuracy ->",preAcc)

# # Naive Bayes Confusion Chart
confusionChart(y_test,y_pred)

#*************************Optimizasyon Sonrası********************************#
print("\nSimulated Annealing Optimizasyon Sonrası : ")
selectedDict = {}

# KNN
modelKNN = KNeighborsClassifier(n_neighbors=9)
modelName = "KNN"

selectedFeatures,bestX,bestAcc = simulatedAnnealing(X, y, modelKNN)
print(f"Simulated Annealing sonrası {modelName} için Accuracy Degeri : {bestAcc}")
# print(selectedFeatures)
print("\n")

# Logistic Regression
modelLR = LogisticRegression()
modelName = "Logistic Regression"

selectedFeatures,bestX,bestAcc = simulatedAnnealing(X, y, modelLR)
print(f"Simulated Annealing sonrası {modelName} için Accuracy Degeri : {bestAcc}")
# print(selectedFeatures)

print("\n")

# Naive Bayes
modelNB = GaussianNB()
modelName = "Naive Bayes"

selectedFeatures,bestX,bestAcc = simulatedAnnealing(X, y, modelNB)
print(f"Simulated Annealing sonrası {modelName} için Accuracy Degeri : {bestAcc}")
# print(selectedFeatures)










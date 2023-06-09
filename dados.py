import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
###############################Para a primeira parte ###################################################
# Carregar o conjunto de dados Iris
iris = sns.load_dataset('iris')
print(tabulate(iris, headers='keys', tablefmt='fancy_grid'))
#Fazer label enconding aos dados
X = iris.drop(['species'], axis=1).values
y = iris['species'].values
def label_encoding(y):
    y_encoded = np.zeros((y.shape[0]))
    for i in range(y.shape[0]):
        if y[i] == "setosa":                
            y_encoded[i] = 1
        else:                
            y_encoded[i] = -1
    return y_encoded
y = label_encoding(y)
X= X[:,0:2]
dataset = np.concatenate((X, y.reshape(-1,1)), axis=1)
df = pd.DataFrame(dataset)
df[2] = df[2].round(0).astype(int) 
df.to_csv('dataset1.csv', index=False, header=False)



##########################################Parte 2########################################################################3
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()
X = iris.data
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
iris = sns.load_dataset('iris')
y = iris['species'].values
def label_encoding(y):
    y_encoded = np.zeros((y.shape[0]))
    for i in range(y.shape[0]):
        if y[i] == "versicolor":                
            y_encoded[i] = 1
        else:                
            y_encoded[i] = -1
    return y_encoded
y = label_encoding(y)
dataset = np.concatenate((X_r, y.reshape(-1,1)), axis=1)
df = pd.DataFrame(dataset)
df[2] = df[2].round(0).astype(int)
df.to_csv('dataset2.csv', index=False, header=False)
print(tabulate(df, headers='keys', tablefmt='fancy_grid'))

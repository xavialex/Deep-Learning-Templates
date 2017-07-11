# Self Organizing Map
# Detección de fraude
# Información sobre el dataset: http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values # SOM es un algoritmo no supervisado, así que y no será utilizada en el entrenamiento

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
# Sigma = radio del BBest matching Unit (BMU) con el que va organizando el resto de nodos
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
# SOM map, en el que aparecen las distancias mean interneural distances (MID)
markers = ['o', 's']
colors = ['r', 'g']
# Los círculos rojos representan gente a la que le han denegado el servicio y los cuadros verdes a los que se le ha concedido
for i, x in enumerate(X):
    # Se itera para i=nº de clientes y x=cada vector cliente para cada iteración
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X) # Diccionario que muestra los clientes asociados a cada nodo
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0) # Se seleccionan los clietnes asociados a los nodos con mayor MID
frauds = sc.inverse_transform(frauds) # Identificación de clientes que cometen fraude
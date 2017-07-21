# Boltzmann Machines
# Algoritmo y teoría descritos en el paper adjunto

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
"""Descripción movies:
    1º Columna: ID de película
    2º Columna: Título
    3º Columna: Género
"""
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
"""Descripción users:
    1º Columna: ID de usuario
    2º Columna: Sexo
    3º Columna: Edad
    4º Columna: Código de identificador de trabajo
    5º Columna: Código
"""
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
"""Descripción ratings:
    1º Columna: ID de usuario
    2º Columna: ID de película
    3º Columna: Calificación
    4º Columna: cuándo se calificó la película
"""

# Preparing the training set and the test set
""" Los sets de datos se han dividido en 5 .base (entrenamiento) y .test a modo
    de k-fold cross validation manual. Aquí se toma sólo un conjunto de datos
"""
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users] # Todos los ID's de películas calificadas por un usuario
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1 # No hay calificación
training_set[training_set == 1] = 0 # El operador OR no está soportado en PyTorch, así que se divide la condición en dos líneas
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv) # Pesos: Tensor que calcula la probabilidad de un nodo oculto respecto a la de un nodo visible
        self.a = torch.randn(1, nh) # Bias para los nodos ocultos: Es necesario el 1 para crear un tensor de 2D para posterior utilización (la primera dimensión para el batch y la segunda para el bias)
        self.b = torch.randn(1, nv) # Bias para los nodos visibles
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t()) # Producto de dos tensores. La .t() es la transpuesta
        activation = wx + self.a.expand_as(wx) # Se usa la función .expand_as() para que el bias se aplique a cada uno de los elementos de mini batch wx
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v) # Se devuelve una muestra de probabilidades de nodos ocultos respecto visibles mediante una distribución de Bernouilli por ser un problema de clasificación binario (la película va a gustar o no)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0) # El cero se usa para conservar la dimensión de tensor
        self.a += torch.sum((ph0 - phk), 0)
nv = len(training_set[0]) # La longitud de la primera línea coincide con el nº de variables
nh = 100 # Nº de variables ocultas a detectar por el modelo (experimental)
batch_size = 100 # Nº de observaciones tras las cuales se aplicará la Contrastive Divergence (experimental)
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. # Contador
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size] # Rango de observaciones desde el usuario actual hasta el actual+batch
        v0 = training_set[id_user:id_user+batch_size]
        ph0, _ = rbm.sample_h(v0)
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0] # Se impide la actualización de los nodos que no tenían calificación
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))
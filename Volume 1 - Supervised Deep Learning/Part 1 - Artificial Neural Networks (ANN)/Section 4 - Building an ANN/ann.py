# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13].values

# Caso particular
new_case = pd.DataFrame.from_items([('CreditScore', [600]), 
                                    ('Geography', ['France']), 
                                    ('Gender', ['Male']), 
                                    ('Age', [40]), 
                                    ('Tenure', [3]), 
                                    ('Balance', [60000]), 
                                    ('NumOfProducts', [2]), 
                                    ('HasCrCard', [1]), 
                                    ('IsActiveMember', [1]), 
                                    ('EstimatedSalary', [50000])])

X = X.append(new_case).values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # Eliminación de la dummy variable, no necesario en Gender ya que la categoría es binaria

new_case = X[10000]
X = np.delete(X, 10000, axis=0)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# La elección de los nodos de una capa puede llegar a ser incluso artística
# Un buen método si se prefiere no experimentar es tomar la media entre los nodos de entrada y salida (11+1)/2=6

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

''' Arquitectura de la red neuronal
        Capas ocultas ReLu y capa de salida Sigmoid
        Esto se debe a que tenemos sólo dos categorías que clasificar (si se queda o no en el banco)
        Por ello la capa de salida tiene una activación sigmoidea y sólo una neurona
        Si hubiera tres o más categorías a clasificar se usaría una activación softmax
'''

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# El optimizer hace referencia ak algoritmo usado para minimizar la función de coste
# adam es un optimizador de gradiente de descenso estocástico (SGD)
# Loss hace referencia a la función de coste. Para casos de clasificación binaria (función de coste logarítmica) se usa binary_crossentropy
# Si hubiera más categorías que clasificar se usaría una category_crossentropy

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

prediction = (classifier.predict(sc.transform(new_case.reshape(1, -1))) > 0.5)

# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
''' Descripción de argumentos
    32 Feature Detectors que generarán 32 convoluciones en la capa convolucional
    (3, 3) Tamaño de filas y columnas de los feature detectors
    (64, 64, 3) Las imágenes sobre las que trabajará tendrán un tamaño de 64x64x3 (respetando la gama RGB)
'''

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# (2, 2) Tamaño del kernel que reducirá los feature maps de la convolutional layer

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())
# Desarrollo de todas las variables de interés tras el proceso de max pooling en un vector que alimente la ANN

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu')) # 128 nodos en la capa oculta
classifier.add(Dense(units = 1, activation = 'sigmoid')) # 1 nodo de salida (clasificación binaria). De haber más de una categoría de salida se usaría la softmax

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
''' Procesamiento de imágenes: Código extraído de la documentación de keras: https://keras.io/preprocessing/image/
    En la documentación de Keras, navegar hasta Image Preprocessing
    Al final del módulo ImageDataGenerator, copiar el código de flow_from_directory
    Este código realiza una serie de transformaciones sobre las imágenes de entrada
    De este modo se consigue obtener mucha más información del mismo dataset
'''

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64), # Tamaño de las imágenes de entrada
                                                 batch_size = 32, # Número de feature maps
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 00:52:14 2022

@author: abigu
"""
import mnist_loader
import network
import pickle
#from keras.models import Sequential
#from keras.layers import Dense

import tensorflow


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data=list(training_data)
test_data=list(test_data)

net=network.Network([784,30,10])

net.SGD(training_data, 20, 10, 3.5, test_data=test_data)

#import torchvision
#import torch

#model = build_model()
#optimizer = torch.optim.SGD(network.Network(), lr=0.01, momentum=0.9)
#optimizer=torch.optim.Adam(network.Network(), lr=0.001)

# define la linea base del modelo
#def baseline_model():
  # crea el modelo
#  model = Sequential()
#  model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal',
#activation='sigmoide'))
#  model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
  # Compila el modelo
#  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#  return model
#y=torch.tensor([one-hot])
#y_hat=torch.tensor([])

#def cross_entropy(y_hat, y):
#    return - torch.log(y_hat[range(len(y_hat)), y])
#cross_entropy(y_hat, y)
#loss = nn.CrossEntropyLoss(reduction='none')
#def net(X):
#    return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)
def softmax(y):
    exps=np.exp(y-np.max(y))
    return exps / np.sum(exps)

def cross_entropy(x, y):
    m=y.shape[0]
    p=softmax(y)
    log_likelihood=-np.log(p[range(m),y])
    loss=np.sum(log_likelihood) / m
    return loss
exit()

archivo=open("red_prueba1.pkl",'wb')
pickle.dump(net, archivo) #dump vaciar de memoria ram a memoria de disco duro
archivo.close()
exit()


#leer el archivo
archivo_lectura=open("red_prueba1.pkl",'rb')
net=pickle.load(archivo_lectura)
archivo_lectura.close()

net.SGD(training_data, 10, 50, 0.5, test_data=test_data)

archivo = open("red_prueba1.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()


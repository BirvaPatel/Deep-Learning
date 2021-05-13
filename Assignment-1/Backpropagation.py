from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import numpy as np


# initialize nwtwork by choosing random weights.
def Initialize_network(tab):
    netwk = list()
    for index_layer in range(1,len(tab)):
        layer = []
        for index_neuron in range(tab[index_layer]):
            randomWeight = []
            for k in range(tab[index_layer-1]+1):
                randomWeight.append(random())
            temp = {'wgts':randomWeight}
            layer.append(temp)
        netwk.append(layer)
    return netwk

#calculating the activation function by equation: sum(weight_i * input_i) + bias
def Activate_fn(wgts, inpts):
	activation = wgts[-1]
	for i in range(len(wgts)-1):
		activation += wgts[i] * inpts[i]
	return activation

#transfering the neuron activation but with sigmoid
#a=activation,derivative=derivative of the parameter
def Trans_sigmoid(a, derivative):
    if derivative == 0:
        return 1.0 / (1.0 + exp(-a))
    else:
        return a * (1.0 - a)

#forward propagate network onutput starting from input
def Frw_propagate(netwk, r, tran_frwd):
    inpts = r
    for layer in netwk:
        n_inputs = []
        for neuron in layer:
            activation = Activate_fn(neuron['wgts'], inpts)
            neuron['output'] = tran_frwd(activation, 0)
            n_inputs.append(neuron['output'])
        inpts = n_inputs
    return inpts

#find the backward propogation error and store it in alpha  
def Frw_propagate_err(netwk, expect, tran_frwd):
        
    for index_layer in reversed(range(len(netwk))):
        layer = netwk[index_layer]
        errors = list()
               
        if index_layer != len(netwk)-1:
            for index_neuron_of_layer_N in range(len(layer)):
                error = 0.0 
                               
                for neuron_layer_M in netwk[index_layer + 1]:
                    error += (neuron_layer_M['wgts'][index_neuron_of_layer_N] * neuron_layer_M['alpha'])
                errors.append(error)
               
        else:
                        
            for index_neuron in range(len(layer)):
                neuron = layer[index_neuron]
                errors.append(expect[index_neuron] - neuron['output'])
                
        for index_neuron in range(len(layer)):
            neuron = layer[index_neuron]
            neuron['alpha'] = errors[index_neuron] * tran_frwd(neuron['output'], 1)

#According to error change the weights
#which is weight += (l_r * error * input)
def Update_weight(netwk, r, l_r):
    for index_layer in range(len(netwk)):
        inpts = r[:-1]
        if index_layer != 0:
                       
            inpts = [neuron['output'] for neuron in netwk[index_layer - 1]]
        for neuron in netwk[index_layer]:
                       
            for index_input in range(len(inpts)):
                neuron['wgts'][index_input] += l_r * neuron['alpha'] * inpts[index_input]
                       
            neuron['wgts'][-1] += l_r * neuron['alpha'] * 1

#to make categorical data more expressive we will use one hot encoding method for output variable.
def one_hot_encoding(o_p, r_dataset):
    expect = [0 for i in range(o_p)]
    expect[r_dataset[-1]] = 1
    return expect

#By using fixed number of epoch train a netwk.
#used stochastic gradient descent.
def Training(netwk, train, test, l_r, epoch, o_p, tran_frwd):
    acc=[]
    for epoch in range(epoch):
        sum_err = 0 
        for r in train:
            outputs = Frw_propagate(netwk, r, tran_frwd)
            expect = one_hot_encoding(o_p, r)
            sum_err += sum([(expect[i]-outputs[i])**2 for i in range(len(expect))])
            Frw_propagate_err(netwk, expect, tran_frwd)
            Update_weight(netwk, r, l_r)
            acc.append(Prediction_acc(netwk, test, tran_frwd))
        accuracies.append(acc)

#Make a prediction with a network
def predict(netwk, r, tran_frwd):
    outputs = Frw_propagate(netwk, r, tran_frwd)
      
    return outputs.index(max(outputs))

#grtting prediction accuracy
def Prediction_acc(netwk, train, tran_frwd):
    pred = list()
    for r in train:
        prediction = predict(netwk, r, tran_frwd)
        pred.append(prediction)
    expec_output = [r[-1] for r in train]
    acc = Acc_matrix(expec_output, pred)
    return acc

#using SGD, implemented the back propogation 
def Back_propagate(train, test, l_r, epoch, hidd, tran_frwd):
    i_p = len(train[0]) - 1
    o_p = len(set([r[-1] for r in train]))
    netwk = Initialize_network([i_p, 5, o_p])
    layerPrint=[]
    for i in range(len(netwk)):
        layerPrint.append(len(netwk[i]))
    Training(netwk, train, test, l_r, epoch, o_p, tran_frwd)
    pred = list()
    for r in test:
        prediction = predict(netwk, r, tran_frwd)
        pred.append(prediction)
    return(pred)

#Load a CSV file
def load_csv(f_name):
    dataset = list()
    with open(f_name, 'r') as file:
        csv_reader = reader(file)
        for r in csv_reader:
            if not r:
                continue
            dataset.append(r)
    return dataset

#chnging the string to float column wise
def str_column_to_float(dataset, column):
    for r in dataset:
        r[column] = float(r[column].strip())
 
#changing string to integer column wise
def str_column_to_int(dataset, column):
    cl_val = [r[column] for r in dataset]
    unic = set(cl_val)
    lookup = dict()
    for i, value in enumerate(unic):
        lookup[value] = i
    for r in dataset:
        r[column] = lookup[r[column]]
    return lookup
 
#min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats
 
# Normalizing columns to the range 0-1 of dataset
def normalize_dataset(dataset, minmax):
    for r in dataset:
        for i in range(len(r)-1):
            r[i] = (r[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
#Split a dataset into k folds using cross validation method
def Cross_valsplit(dataset, fold):
    ds_spt = list()
    ds_copy = list(dataset)
    size_of_fold = int(len(dataset) / fold)
    for i in range(fold):
        fold = list()
        while len(fold) < size_of_fold:
            index = randrange(len(ds_copy))
            fold.append(ds_copy.pop(index))
        ds_spt.append(fold)
    return ds_spt
 
#count accuracy percentage with predicted and actual value
def Acc_matrix(actl, prdctd):
    correct = 0
    for i in range(len(actl)):
        if actl[i] == prdctd[i]:
            correct += 1
    return correct / float(len(actl)) * 100.0
 
#Evaluation of an algorithm using a cross validation split
def Evaluate_algo(dataset, algorithm, fold, *args):
       
    folds = Cross_valsplit(dataset, fold)
    final_ans = list()
    for fold in folds:
            
        train_set = list(folds)      
        train_set.remove(fold)       
        train_set = sum(train_set, [])
        
        test_set = list()
        for r in fold:               
            r_copy = list(r)
            test_set.append(r_copy)
            r_copy[-1] = None
               
        prdctd = algorithm(train_set, fold, *args)

        actl = [r[-1] for r in fold]
        
        acc = Acc_matrix(actl, prdctd)
        final_ans.append(acc)
      
        
    return final_ans

seed(1)
accuracies = list()
f_name = 'iris.csv'
dataset = load_csv(f_name)
#replacing string to floats
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
	
#replacing column to integers
str_column_to_int(dataset, len(dataset[0])-1)

minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

fold = 5
l_r = 0.3
epoch = 50
hidd = 4

final_ans = Evaluate_algo(dataset, Back_propagate, fold, l_r, epoch, hidd, Trans_sigmoid)
print('Testing acc: %.3f%%' % (sum(final_ans)/float(len(final_ans))))

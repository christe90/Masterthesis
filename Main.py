import tensorflow as tf
import datetime
import os
import math
import random
import itertools
import numpy as np
import pandas as pd
import inspect

from tensorflow import keras

import DHNNClasses as dhnn
import matlab.engine
#######################################
def realFunctionBalken(**x_Data):
    y_Data = ((1/3) *x_Data['P'] * x_Data['L'] ** 3 / (x_Data['E'] * x_Data['I']))
    return y_Data
def realFunctionBewegungV0(**x_Data):
    y_Data = (x_Data['v0'] * x_Data['t']) - (0.5)* (x_Data['g'] * x_Data['t']**2)
    return y_Data
def realPendel(**x_Data):
    y_Data = (x_Data['y0'] * math.cos(math.sqrt(x_Data['D'] / x_Data['m']) * x_Data['t']))
    return y_Data

###################################
generatedData = dhnn.TestCaseGenerator(knownFunction= realPendel,
                            numberOfDatapoints= 100,
                            noiseLevel=0.0, 
                            inputVariables= ['y0','t','D','m'], 
                            outputVariable= 'y')
matlabEng = matlab.engine.start_matlab()
zeit = (datetime.datetime.now()).strftime("%m%d%Y%H%M%S")
#######################################
matlabEng = matlab.engine.start_matlab()
zeit = (datetime.datetime.now()).strftime("%m%d%Y%H%M%S")
#######################################
Versuch = 'SchwingungFederpendel'
VersuchsOrdner = './02_Projekt/Versuche/' + Versuch 
if not os.path.exists(VersuchsOrdner):
    os.makedirs(VersuchsOrdner)
yName = 'y'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=(VersuchsOrdner + '/logs/'+ zeit +'/'),
                                                         histogram_freq=1)
generatedData.Dataset.to_csv((VersuchsOrdner + '/Data.csv'), index=False, sep=';')
data = pd.read_csv((VersuchsOrdner + '/Data.csv'),sep=';')
dimensionsMatrix = pd.read_csv((VersuchsOrdner + '/Dimensionsmatrix.csv'),sep=';',index_col = 0)
topology = pd.read_csv((VersuchsOrdner + '/Topologie.csv'),sep=';')
splitRatio=(0.6,0.2,0.2)
myLoss = 'MSE'
myMetrics = ['MAPE','MSE']
myCallbacks = [tensorboard_callback, dhnn.MyCallback(log_dir = VersuchsOrdner,matlabEng = matlabEng),tf.keras.callbacks.TerminateOnNaN()]
summaryWriter = tf.summary.create_file_writer(VersuchsOrdner+"/mylogs/eager/"+ zeit+'/')
maxEpochs = 1000
myOptimizer = 'Adadelta'
##########################

newDataset = dhnn.Dataset(dimensionsMatrix= dimensionsMatrix,
                    data = data, 
                    splitRatio= splitRatio, 
                    outputName= yName,
                    factor = 5)

myModel  = dhnn.MyModel(topology,summaryWriter, newDataset.piMatrix)

myModel.compile(
    loss= myLoss,
    optimizer = myOptimizer,
    metrics = myMetrics, 
    )

myModel.model_graph()

tf.keras.utils.plot_model(
    myModel.model_graph(),                   
    to_file=(VersuchsOrdner + '/modelplot.png'), dpi=96,            
    show_shapes=True, show_layer_names=False,
    expand_nested=True, rankdir='LR'                     
    )

myModel.fit(x=newDataset.trainingData['x_Values'], 
          y=newDataset.trainingData['y_Values'], 
          epochs=maxEpochs,
          batch_size = 5,
          validation_data=(newDataset.testData['x_Values'], newDataset.testData['y_Values']),
          callbacks = myCallbacks)

myModel.save(VersuchsOrdner)

myFormula = dhnn.MyModelInterpretation(myModel)
myFormula.get_formula()
with open((VersuchsOrdner + '/weights.txt'), "a") as myfile:
    print(myFormula.layerFormulas['FinalLayer'], file= myfile)

myMessenger = dhnn.MsTeamsMessenger()
myMessenger('Versuch ' + Versuch+ ' ist fertig. \n Nach '+ str(len(myModel.history.epoch)) + ' Epochen mit einem Loss von ' + str(myModel.history.history['loss'][-1]))
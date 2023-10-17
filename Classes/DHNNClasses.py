import tensorflow as tf
import datetime
import os
import math
import random
import itertools
import numpy as np
import pandas as pd
import inspect
from fractions import Fraction
from tensorflow import keras

class Dataset(): 
    def __init__(self,dimensionalMatrix, data, splitRatio, y_Name, augmentationFactor):
        self.data = data
        self.splitRatio = splitRatio
        self.y_Name = y_Name
        self.augmentationFactor = augmentationFactor
        self.dimensionsMatrix = dimensionalMatrix
        self._getDimensionlessVariables()
        self._getDimensionalVariables()
        self.getPiMatrices()
        self.piMatrix = self.piMatrices['matrix'][0]
        self.getAugmentedData()
        self.splitData()
    
    def getPiMatrices(self):
        self.piMatrices = []
        ## Teil der PiMatrix aus Dimensionslosen Größen
        self.piMatrixDimensionless = np.eye(len(self.dimensionlessVariables))
        ## Teil der PiMatrix aus Dimensionsbehafteten Größen
        self.piMatricesDimensional = []
        ## Alle möglichen abgeleiteten Dimensionslosen Gruppen aus der Dimensionsmatrix bestimmen
        self._getDimensionlessGroups()
        ## Die beiden Matrizen jeweils zusammenfügen und ordnen, dass die gesuchte Größe an der ersten Stelle steht
        self._orderDimensionsMatrices()
        ## Die gefundenen PiMatrizen in einen Dataframe konvertieren und mit absteigendem Counter sortieren
        self.piMatrices = pd.DataFrame(self.piMatrices, columns=['matrix','counter'])
        self.piMatrices = self.piMatrices.sort_values(['counter'])
    
    def getAugmentedData(self):
        dataset_new = []
        ## über alle realen Datenpunkte iterieren
        for i in range(0,len(self.data)):
            ## choose a random possible pi matrix
            piMatrix = self.piMatrices['matrix'][int(round(random.uniform(0,len(self.piMatrices)-1)))]
            ## realen Datenpunkt aus realem Datenset entnehmen und umsortieren
            real_datapoint = self.data.iloc[i,:].reindex(piMatrix.columns)
            ## vollähnliche Datenpunkte erstellen
            augmented_Datapoints = self._getAugmentedDatapoints(real_datapoint, piMatrix)
            ## erstellte vollähnliche Datenpunkt sortieren und der Gesamtliste hinzufügen
            augmented_Datapoints = augmented_Datapoints.reindex(columns = self.data.columns)
            augmented_Datapoints = augmented_Datapoints.to_numpy()
            dataset_new.extend(augmented_Datapoints.tolist())
        dataset_new = pd.DataFrame(dataset_new, columns = self.data.columns)
        self.augmentedData = dataset_new
        ## self._scaleAugmentedData(dataset_new)    

    def splitData(self):
        size = self.augmentedData.shape[0]
        data = self.augmentedData.sample(frac = 1) #Shuffle Dataset
        # Bestimmung der Anzahl der Trainings-, Validierungs- und Testdaten
        sizeTrainingData = math.floor(size * self.splitRatio['train']) 
        sizeValidationData = math.floor(size * self.splitRatio['val']) 
        #Split der Daten in Trainings-, Validierungs- und Testdaten
        trainingData = data[:sizeTrainingData:]
        validationData = data[sizeTrainingData:(sizeTrainingData + sizeValidationData):]
        testData = data[(sizeTrainingData + sizeValidationData)::]
        #Aufteilung in X und Y Werte und Konvertierung in Numpy-Arrays zur Weiterverarbeitung in TF
        x_Dimensions = list(self.data.columns)
        x_Dimensions.remove(self.y_Name)
        self.trainingData = {'x_Values': trainingData[x_Dimensions].to_numpy(), 'y_Values':trainingData[self.y_Name].to_numpy()}
        self.validationData = {'x_Values': validationData[x_Dimensions].to_numpy(), 'y_Values':validationData[self.y_Name].to_numpy()}
        self.testData = {'x_Values': testData[x_Dimensions].to_numpy(), 'y_Values':testData[self.y_Name].to_numpy()}

    def _orderDimensionsMatrices(self):
        x_Dimensions = list(self.dimensionsMatrix.columns)
        x_Dimensions.remove(self.y_Name)
        for piMatrix in self.piMatricesDimensional:
            dimMatrix = piMatrix[0].to_numpy()
            noDimMatrix = self.piMatrixDimensionless
            new_Matrix = np.block([ [dimMatrix, np.zeros(shape = (dimMatrix.shape[0], len(self.dimensionlessVariables)))],
                                    [np.zeros(shape = (len(self.dimensionlessVariables), dimMatrix.shape[1])), noDimMatrix] ])
            new_Matrix = pd.DataFrame(data = new_Matrix, columns = list(piMatrix[0].columns) + self.dimensionlessVariables)
            new_Matrix = new_Matrix.reindex(columns = (list(self.y_Name) + x_Dimensions))
            new_Matrix = new_Matrix.sort_values(by=list(new_Matrix.columns)[0:new_Matrix.shape[0]], ascending=False)
            self.piMatrices.append([new_Matrix, piMatrix[1]])

    def _getDimensionlessVariables(self):
        self.dimensionlessVariables = []
        for variable in self.dimensionsMatrix.columns:
            if not self.dimensionsMatrix[variable].any():
                self.dimensionlessVariables.append(variable)
    
    def _getDimensionalVariables(self):
        self.dimensionalVariables = []
        for variable in self.dimensionsMatrix.columns:
            if self.dimensionsMatrix[variable].any():
                self.dimensionalVariables.append(variable)

    def _getDimensionlessGroups(self):
        self._getNumberOfDimensionlessGroups()
        possibleCombinations = self._getPossibleCombinations()
        for possibleCombination in possibleCombinations:
            ## Auf Stufenform bringen
            A, B = self._getAnBMatrices(possibleCombination)
            ## Checken ob B Singulär ist
            if(np.linalg.det(B) != 0):
                ## PiMatrix berechnen
                piMatrix = self._createPiMatrix(A, B, possibleCombination)
                ## Anzahl der Spalten nicht ganzzahligen Werten ermitteln
                floatColumnsCount = self._countColumnsWithFloat(piMatrix)
                ## gefundene Matrix an bestehende Liste anhängen
                self.piMatricesDimensional.append([ piMatrix, floatColumnsCount])
    
    def _getNumberOfDimensionlessGroups(self):
        arrayMatrix = self.dimensionsMatrix[self.dimensionalVariables].to_numpy()
        self.rankMatrix = np.linalg.matrix_rank(arrayMatrix)
        self.numberOfDimensionlessGroups = len(self.dimensionalVariables) - self.rankMatrix

    def _getPossibleCombinations(self):
        all_Dimensions = set(self.dimensionalVariables)
        # Entferne alle Dimensionslosen Größen und Y aus X
        if self.y_Name in self.dimensionalVariables:
            x_Dimensions = list(set(self.dimensionalVariables).difference(set(self.y_Name)))
        else:
            x_Dimensions = all_Dimensions
        possibleCombinations = list(itertools.permutations(x_Dimensions,len(x_Dimensions)))        
        return possibleCombinations
    
    def _getAnBMatrices(self, possibleCombination):
        ## Dimensionsmatrix umsortieren
        if self.y_Name in self.dimensionalVariables:
            dimensionsMatrix_new = self.dimensionsMatrix[self.dimensionalVariables].reindex(columns = (list(self.y_Name) + list(possibleCombination)))
        else:
            dimensionsMatrix_new = self.dimensionsMatrix[self.dimensionalVariables].reindex(columns = ( list(possibleCombination)))
        ## Dimensionslose Produkte ermitteln
        arrayMatrix = dimensionsMatrix_new.to_numpy()
        arrayMatrix = np.linalg.qr(arrayMatrix)[1] # Q/R Zerlegung
        arrayMatrix = arrayMatrix[:self.rankMatrix,:] ## Entfernen von nicht linear unabhängigen Zeilen
        arrayMatrix[0,:] = arrayMatrix[0,:] /  arrayMatrix[0,0] ## die gesucht größe mit auf 1 normieren
        A, B = np.split(arrayMatrix,[self.numberOfDimensionlessGroups],1) ##Splitten in Untermatrizen
        return A, B

    def _createPiMatrix(self, A, B, possibleCombination):
        k1 = np.eye(self.numberOfDimensionlessGroups)
        k2 = -(1) * np.matmul(np.linalg.inv(B) ,np.matmul(A,k1))
        if self.y_Name in self.dimensionalVariables:
            dimensionlessGroup = pd.DataFrame(data=(np.concatenate((k1, np.transpose(k2)), axis=1)) , columns = (list(self.y_Name) + list(possibleCombination)))
        else:
            dimensionlessGroup = pd.DataFrame(data=(np.concatenate((k1, np.transpose(k2)), axis=1)) , columns = (list(possibleCombination)))
        return dimensionlessGroup

    def _countColumnsWithFloat(self, piMatrix):
        counter = 0
        for column in piMatrix.columns:
            if ((piMatrix[column] % 1  != 0).all()):
                counter = counter + 1
        return counter

    def _scaleAugmentedData(self, augmentedData):
        augmentedData_scaled = np.zeros_like(augmentedData.to_numpy())
        augmentedArray = augmentedData.to_numpy()
        rawArray = self.data.to_numpy()
        for i in range(0,len(augmentedData.columns)):
            minAugmented = np.amin(augmentedArray[:,i])
            maxAugmented = np.amax(augmentedArray[:,i])
            minRaw = np.amin(rawArray[:,i])
            maxRaw = np.amax(rawArray[:,i])
            augmentedData_scaled[:,i] = (augmentedArray[:,i] - minAugmented) / (maxAugmented - minAugmented) * (maxRaw - minRaw) + minRaw
        augmentedData_scaled = pd.DataFrame(augmentedData_scaled, columns = augmentedData.columns)
        return augmentedData_scaled

    def _getAugmentedDatapoints(self, real_datapoint, piMatrix):
        augmented_Datapoints = []
        ##Datapoint und PiMatrix in Numpy-arrays konvertieren
        real_datapoint = real_datapoint.to_numpy()
        piMatrixColumns = piMatrix.columns
        piMatrix = piMatrix.to_numpy()
        ## den realen Datenpunkt der neuen Liste hinzufügen
        augmented_Datapoints.append(real_datapoint.tolist())
        ## zufällige auswahl eines zu ändernden Wertes im Datenpunkt
        change_column = random.choice(range(len(piMatrix),len(real_datapoint)))
        ## minima und maxima der zufälligen dimension in den originaldaten bestimmen
        rawData = self.data.copy().reindex(columns = piMatrixColumns)
        rawArray = rawData.to_numpy()
        orig_max = np.amax(rawArray[:,change_column])
        orig_min = np.amin(rawArray[:,change_column])
        for i in range(0,self.augmentationFactor - 1):
            datapoint_new = real_datapoint.copy()
            while True:
                randomValue = random.gauss(real_datapoint[change_column], 1)
                # Vorzeichen des zufälligen wertes dem usprünglichen wert gleichsetzen
                randomValue = np.abs(randomValue) * real_datapoint[change_column] / np.abs(real_datapoint[change_column])
                if (randomValue > orig_min and randomValue < orig_max):
                    break
            datapoint_new[change_column] = randomValue
            ## Die Werte der anderen Dimensionen bei beibehaltung der Pi's berechnen
            for j in range(0,len(piMatrix)):
                invariante = np.prod(np.power(real_datapoint, piMatrix[j]))
                datapoint_new[j] = invariante / (np.prod(np.power(datapoint_new[len(piMatrix):],piMatrix[j][len(piMatrix):])))
            ## Den neu erstellten Datenpunkt der neuen Liste hinzufügen
            augmented_Datapoints.append(datapoint_new.tolist())
        augmented_Datapoints = pd.DataFrame(augmented_Datapoints, columns = piMatrixColumns)
        return augmented_Datapoints

class MyModel(tf.keras.Model):
    def __init__(self,layerTopology, summaryWriter, piMatrix = None, dimensionless = True):
        super(MyModel,self).__init__()
        self.piMatrix = piMatrix
        ##self.current_epoch = tf.Variable(tf.random.uniform((1,1)), shape=(1,1))
        self.summaryWriter = summaryWriter
        self.layerTopology = layerTopology
        self._createLayers()
        self.dimensionless = dimensionless

    def call(self,input_tensor):
        y = []
        y.append(self.piLayer(input_tensor))
        y.append(self.shortcutLayer(input_tensor))
        for j in range(0,len(self.layerTopology.index)):
            if self.layerTopology['Type'][j] == 'functionalBlock':
                createdLayer = list(filter(lambda layer: layer.blockname == self.layerTopology['Name'][j], self.createdLayers))[0]
                input = int(self.layerTopology['Input'][j])
                y.append(createdLayer(y[input]))
            elif self.layerTopology['Type'][j] == 'concat':
                inputList = []
                inputRaw = self.layerTopology['Input'][j].split(',')
                for j2 in range(0,len(inputRaw)):
                    inputNumber = int(inputRaw[j2])
                    inputList.append(y[inputNumber])
                y.append(tf.keras.layers.concatenate(inputList))
            elif self.layerTopology['Type'][j] == 'finalLayer':
                input = int(self.layerTopology['Input'][j])
                y.append(self.finalLayer(y[input]))
            elif self.layerTopology['Type'][j] == 'multiply':
                inputList = []
                inputRaw = self.layerTopology['Input'][j].split(',')
                for j2 in range(0,len(inputRaw)):
                    inputNumber = int(inputRaw[j2])
                    inputList.append(y[inputNumber])
                y.append(tf.keras.layers.multiply(inputList))
        return y[-1]

    def build(self, input_shape):
        super(MyModel,self).build(input_shape)
        if self.piMatrix is not None:
            self._getWeights()
            self._setWeights()
        
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        ## Falls loglevel = 3 dann gradienten und Eingangsdaten und label loggen
        if self.logLevel == 3:
          self._logGradients(gradients, x, y)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}       

    def model_graph(self):
        x = keras.Input(shape=(self.piMatrix.shape[-1] -1))
        return keras.Model(inputs=[x], outputs=self.call(x))

    def set_log_level(self, logLevel, log_path):
        self.logLevel = logLevel
        self.log_path = log_path
    
    def _getWeights(self):
        self.piLayerWeight = np.transpose(self.piMatrix.values[1:,1:])
        self.shortCutLayerWeight = (self.piMatrix.values[0,1:]).reshape(-1, 1) * (-1)
    
    def _setWeights(self):
        if self.dimensionless == True:
            self.piLayer.set_weights([self.piLayerWeight])
            self.piLayer.trainable = False
            self.shortcutLayer.set_weights([self.shortCutLayerWeight])
            self.shortcutLayer.trainable = False

    def _logGradients(self, gradients, x, y):
        tf.print(gradients,output_stream = ("file://"+self.log_path+ "/gradients.txt"),summarize=-1)
        with self.summaryWriter.as_default():
            tf.summary.histogram("X-Data", x,step = 1)
            tf.summary.histogram("Y-Data", y,step = 1)
            for gradient in gradients:
                tf.summary.histogram('Grad', gradient,step = 1)

    def _createLayers(self):
        self.createdLayers = []
        for j in range(2,len(self.layerTopology.index) -1 ):
            if self.layerTopology['Type'][j] == 'functionalBlock':
                newLayer = MyFunctionBlock(blockname = self.layerTopology['Name'][j], fn_type= self.layerTopology['fn'][j])
                self.createdLayers.append(newLayer)
        self.piLayer = MyProductBlock('piLayer',self.piMatrix.shape[0]-1)
        self.shortcutLayer = MyProductBlock('shortcutLayer')
        self.finalLayer = keras.layers.Dense(1, name ='FinalLayer', use_bias = False)

class MyBlock(tf.keras.layers.Layer):
    def __init__(self, blockName, size = 1, bias = False):
        super(MyBlock, self).__init__()
        self.blockName = blockName
        self.bias = bias
        self.myLayer = keras.layers.Dense(size,use_bias = self.bias)

    def call(self, input_tensor):
        x = input_tensor
        x = self.myLayer(x)
        return x

class MyFunctionBlockFull(tf.keras.layers.Layer):
    def __init__(self, blockname, bias = False):
        super(MyFunctionBlockFull, self).__init__()
        self.blockname = blockname
        self.SumBlock = MyBlock(self.blockname + '_Sum',bias)
        self.ProductBlock  = MyProductBlock(self.blockname + '_Product', bias)

    def call(self,input_tensor):
        x = input_tensor
        x_sin = tf.map_fn(tf.math.sin, x)
        x_cos = tf.map_fn(tf.math.cos, x)
        x_tanh = tf.map_fn(tf.math.tanh, x)
        x_exp = tf.map_fn(tf.math.exp, x)
        x_ln = tf.map_fn(tf.math.abs, x)
        x_ln = tf.map_fn(tf.math.log, x_ln)
        x_sigmoid = tf.map_fn(tf.math.sigmoid, x)
        x = keras.layers.concatenate([x, x_sin, x_cos, x_tanh, x_exp, x_ln, x_sigmoid])
        x_Sum = self.SumBlock(x)
        x_Prod = self.ProductBlock(x)
        x = keras.layers.concatenate([x_Sum, x_Prod])
        return x   

class MyProductBlock(tf.keras.layers.Layer):
    def __init__(self, blockname,size = 1, bias = False):
        super(MyProductBlock, self).__init__()
        self.blockname = blockname
        self.bias = bias
        self.SumBlock = MyBlock(self.blockname + '_Sum',size, self.bias)

    def call(self,input_tensor):
        x = input_tensor
        x = tf.map_fn(tf.math.abs, x)
        x = tf.map_fn(tf.math.log, x)
        x = self.SumBlock(x)
        x = tf.map_fn(tf.math.exp, x)
        return x

class MyFunctionBlockSeriell(tf.keras.layers.Layer):
    def __init__(self, blockname):
        super(MyFunctionBlockSeriell, self).__init__()
        self.blockname = blockname
        self.ProductBlock  = MyProductBlock(self.blockname + '_Product', bias = True)
        self.SumBlock = MyBlock(self.blockname + '_Sum', bias = True)
        self.DropLn = tf.keras.layers.Dense(1, activation='relu', use_bias = False ,kernel_initializer='zeros')
        self.SumBlockLN = MyBlock(self.blockname + '_SumLN')
        self.DropCos = tf.keras.layers.Dense(1, activation='relu', use_bias = False,kernel_initializer='zeros')
        self.SumBlockCOS = MyBlock(self.blockname + '_SumCOS')
        self.DropSIN = tf.keras.layers.Dense(1, activation='relu', use_bias = False,kernel_initializer='zeros')
        self.SumBlockSIN = MyBlock(self.blockname + '_SumSIN')
        self.DropTANH = tf.keras.layers.Dense(1, activation='relu', use_bias = False,kernel_initializer='zeros')
        self.SumBlockTANH= MyBlock(self.blockname + '_SumTANH')
        self.DropE = tf.keras.layers.Dense(1, activation='relu', use_bias = False,kernel_initializer='zeros')
        self.SumBlockE = MyBlock(self.blockname + '_SumE')
    
    def call(self, input_tensor):
        x_Prod = self.ProductBlock(input_tensor)
        inputSum = keras.layers.concatenate([input_tensor, x_Prod])
        x_Sum = self.SumBlock(inputSum)
        ##ln
        input_ln = tf.map_fn(tf.math.abs, x_Sum)
        input_ln = tf.map_fn(tf.math.log, input_ln)
        input_ln = self.DropLn(input_ln)
        input_ln = keras.layers.concatenate([input_ln, x_Sum])
        x_ln = self.SumBlockLN(input_ln)
        ##sin
        input_sin = tf.map_fn(tf.math.sin, x_ln)
        input_sin = self.DropSIN(input_sin)
        input_sin = keras.layers.concatenate([input_sin, x_ln])
        x_sin = self.SumBlockSIN(input_sin)
        ##cos
        input_cos = tf.map_fn(tf.math.cos, x_sin)      
        input_cos = self.DropCos(input_cos)
        input_cos = keras.layers.concatenate([input_cos, x_sin])
        x_cos = self.SumBlockCOS(input_cos)
        ## e
        input_e = tf.map_fn(tf.math.exp, x_cos)
        input_e = self.DropE(input_e)
        input_e = keras.layers.concatenate([input_e, x_cos])
        x_e = self.SumBlockE(input_e)
        ## tanh
        input_tanh = tf.map_fn(tf.math.tanh, x_e)
        input_tanh = self.DropTANH(input_tanh)
        input_tanh = keras.layers.concatenate([input_tanh, x_e])
        x_tanh = self.SumBlockTANH(input_tanh)
        return x_tanh

class MyFunctionBlock(tf.keras.layers.Layer):
    def __init__(self, blockname, bias = False, fn_type = 'lin'):
        super(MyFunctionBlock, self).__init__()
        self.fn_type = fn_type
        self.blockname = blockname
        self.ProductBlock  = MyProductBlock(self.blockname + '_Product', bias = False)
        self.SumBlock = MyBlock(self.blockname + '_Sum', bias = True)

    def call(self,input_tensor):
        x = input_tensor
        if self.fn_type == 'sin':
            x_fn = tf.map_fn(tf.math.sin, x)
            x = keras.layers.concatenate([x, x_fn])
        elif self.fn_type == 'cos':
            x_fn = tf.map_fn(tf.math.cos, x)
            x = keras.layers.concatenate([x, x_fn])
        elif self.fn_type == 'tanh':
            x_fn = tf.map_fn(tf.math.tanh, x)
            x = keras.layers.concatenate([x, x_fn])
        elif self.fn_type == 'exp':
            x_fn = tf.map_fn(tf.math.exp, x)
            x = keras.layers.concatenate([x, x_fn])
        elif self.fn_type == 'log':
            x_fn = tf.map_fn(tf.math.abs, x)
            x_fn = tf.map_fn(tf.math.log, x_fn)
            x = keras.layers.concatenate([x, x_fn])
        elif self.fn_type == 'sig':
            x_fn = tf.map_fn(tf.math.sigmoid, x)
            x = keras.layers.concatenate([x, x_fn])
        elif self.fn_type == 'lin':
            x = x 
        x_Sum = self.SumBlock(x)
        x_Prod = self.ProductBlock(x)
        x = keras.layers.concatenate([x_Sum, x_Prod])
        return x
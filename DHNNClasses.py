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
    def splitData(self, data, splitRatio, size, outputName):
        data = data.sample(frac = 1) #Shuffle Dataset
        # Bestimmung der Anzahl der Trainings-, Validierungs- und Testdaten
        sizeTrainingData = math.floor(size * splitRatio[0]) 
        sizeValidationData = math.floor(size * splitRatio[1]) 
        #Split der Daten in Trainings-, Validierungs- und Testdaten
        trainingData = data[:sizeTrainingData:]
        validationData = data[sizeTrainingData:(sizeTrainingData + sizeValidationData):]
        testData = data[(sizeTrainingData + sizeValidationData)::]
        #Aufteilung in X und Y Werte und Konvertierung in Numpy-Arrays zur Weiterverarbeitung in TF
        x_Dimensions = list(data.columns)
        x_Dimensions.remove(outputName)
        trainingData = {'x_Values': trainingData[x_Dimensions].to_numpy(), 'y_Values':trainingData[outputName].to_numpy()}
        validationData = {'x_Values': validationData[x_Dimensions].to_numpy(), 'y_Values':validationData[outputName].to_numpy()}
        testData = {'x_Values': testData[x_Dimensions].to_numpy(), 'y_Values':testData[outputName].to_numpy()}
        return trainingData, validationData, testData

    def orderDimensionsMatrix(self, dimensionsMatrix, outputName):
        ##dimensionsMatrix = pd.DataFrame(variablesDimensions).fillna(0) ## Liste der Dimensionen in Matrix überführen
        ## Spalten der Dimensionsmatrix vertauschen, sodass die gesuchte Groesse an erster Stelle steht
        x_Dimensions = list(dimensionsMatrix.columns)
        x_Dimensions.remove(outputName)
        dimensionsMatrix = dimensionsMatrix.reindex(columns = (list(outputName) + x_Dimensions))
        return dimensionsMatrix

    def getDimensionlessGroups(self, dimensionsMatrix, outputName):
        x_Dimensions = list(dimensionsMatrix.columns)[1:] ## Die Eingangvariablen (ohne den Ausgangswert an Stelle 0)
        ## Ermittlung der Anzahl der dimensionslosen Gruppen
        arrayMatrix = dimensionsMatrix.to_numpy()
        rankMatrix = np.linalg.matrix_rank(arrayMatrix)
        numberOfDimensionlessGroups = len(dimensionsMatrix.columns) - rankMatrix
        ## Ermittlung aller möglichen Kombinationen der Eingangsvariablen
        possibleCombinations = list(itertools.permutations(x_Dimensions,len(x_Dimensions)))
        possiblePiMatrix = []
        for inputCombination in possibleCombinations:
            ## Dimensionsmatrix umsortieren
            dimensionsMatrix = dimensionsMatrix.reindex(columns = (list(outputName) + list(inputCombination)))
            ## Dimensionslose Produkte ermitteln
            arrayMatrix = dimensionsMatrix.to_numpy()
            arrayMatrix = np.linalg.qr(arrayMatrix)[1] # Q/R Zerlegung
            arrayMatrix = arrayMatrix[:rankMatrix,:] ## Entfernen von nicht linear unabhängigen Zeilen
            arrayMatrix[0,:] = arrayMatrix[0,:] /  arrayMatrix[0,0] ## die gesucht größe mit auf 1 normieren
            A, B = np.split(arrayMatrix,[numberOfDimensionlessGroups],1) ##Splitten in Untermatrizen
            ## checken ob B nicht singulär ist
            if(np.linalg.det(B) != 0):
                k1 = np.eye(numberOfDimensionlessGroups)
                k2 = -(1) * np.matmul(np.linalg.inv(B) ,np.matmul(A,k1))
                dimensionlessGroups_temp = pd.DataFrame(data=(np.concatenate((k1, np.transpose(k2)), axis=1)) , columns = dimensionsMatrix.columns)                
                ## Anzahl der Spalten mit floatingPoints ermitteln
                counter = 0
                for column in dimensionlessGroups_temp.columns:
                    if ((dimensionlessGroups_temp[column] % 1  != 0).all()):
                        counter = counter + 1
                possiblePiMatrix.append([ dimensionlessGroups_temp, counter])
        possiblePiMatrix = pd.DataFrame(possiblePiMatrix, columns=['matrix','counter'])
        possiblePiMatrix = possiblePiMatrix.sort_values(['counter'])
        return possiblePiMatrix

    def scaleAugmentedData(self, rawData, augmentedData):
        augmentedData_scaled = np.zeros_like(augmentedData.to_numpy())
        i = 0
        augmentedArray = augmentedData.to_numpy()
        rawArray = rawData.to_numpy()
        for i in range(0,len(augmentedData.columns)):
            minAugmented = np.amin(augmentedArray[:,i])
            maxAugmented = np.amax(augmentedArray[:,i])
            minRaw = np.amin(rawArray[:,i])
            maxRaw = np.amax(rawArray[:,i])
            augmentedData_scaled[:,i] = (augmentedArray[:,i] - minAugmented) / (maxAugmented - minAugmented) * (maxRaw - minRaw) + minRaw
        augmentedData_scaled = pd.DataFrame(augmentedData_scaled, columns = augmentedData.columns)
        return augmentedData_scaled
    def getAugmentedData(self, data_raw, possiblePiMatrixes, factor):
        ## for schleife über bisherige Werte
        dataset_new = []
        data_raw_orig = data_raw.copy()
        for i in range(0,len(data_raw)):
            dataset_new_temp = []
            ## choose pi matrix
            piMatrix = possiblePiMatrixes[int(round(random.uniform(1,len(possiblePiMatrixes)-1)))]
            ## datapoint an pimatrix anpassen
            data_raw = data_raw.reindex(columns = piMatrix.columns)
            ## zufällige auswahl eines zu ändernden Wertes im Datenpunkt
            datapoint = data_raw.to_numpy()[i]
            pimatrix = piMatrix.to_numpy()
            dataset_new_temp.append(datapoint)
            change_column = random.choice(range(len(pimatrix),len(datapoint)))
            for i2 in range(0,factor-1):
                datapoint_new = datapoint.copy()
                datapoint_new[change_column] = random.gauss(datapoint[change_column], 1)
                # Vorzeichen des zufälligen wertes dem usprünglichen wert gleichsetzen
                datapoint_new[change_column] = np.abs(datapoint_new[change_column]) * datapoint[change_column] / np.abs(datapoint[change_column])
                #
                for j in range(0,len(pimatrix)):
                    invariante = np.prod(np.power(datapoint, pimatrix[j]))
                    datapoint_new[j] = invariante / (np.prod(np.power(datapoint_new[len(pimatrix):],pimatrix[j][len(pimatrix):])))
                datapoint_new = datapoint_new.tolist()
                dataset_new_temp.append(datapoint_new)
            dataset_new_temp = pd.DataFrame(dataset_new_temp, columns = piMatrix.columns)
            dataset_new_temp = dataset_new_temp.reindex(columns = data_raw_orig.columns)
            dataset_new_temp = dataset_new_temp.to_numpy()
            dataset_new.extend(dataset_new_temp)
        augmentedData = pd.DataFrame(dataset_new, columns = data_raw_orig.columns)
        augmentedData_scaled = self.scaleAugmentedData( data_raw, augmentedData)
        return augmentedData_scaled
        
    def __init__(self,dimensionsMatrix, data, splitRatio, outputName, factor = 1):
        self.data = pd.DataFrame(data)
        self.splitRatio = splitRatio
        self.outputName = outputName
        self.dimensionsMatrix = dimensionsMatrix
        self.dimensionsMatrix = self.orderDimensionsMatrix(self.dimensionsMatrix, self.outputName)
        self.possiblePiMatrix = self.getDimensionlessGroups(self.dimensionsMatrix, self.outputName)        
        self.piMatrix = self.possiblePiMatrix['matrix'][0]
        self.data = self.data.reindex(columns = self.piMatrix.columns)
        self.data = self.getAugmentedData(self.data, self.possiblePiMatrix['matrix'], factor)
        self.trainingData, self.validationData, self.testData = self.splitData(self.data,self.splitRatio, self.data.shape[0], self.outputName)

class TestCaseGenerator():
    def addNoise(self, dataset, noiseLevel):
        noiseSet = pd.DataFrame(1 + np.random.normal(0, noiseLevel, dataset.shape),columns = dataset.columns)
        dataset = dataset * noiseSet
        return dataset

    def createDataPoint(self, knownFunction, inputVariables, outputVariable):
        dataPoint = {}
        for variable in inputVariables:
            dataPoint[variable] = np.random.randint(1,10)
        dataPoint[outputVariable] = knownFunction(**dataPoint)
        return dataPoint

    def createDataset(self, knownFunction, numberOfDatapoints, noiseLevel, inputVariables, outputVariable):
        dataset = []
        for i in range(0,numberOfDatapoints):
            datapoint = self.createDataPoint(knownFunction, inputVariables, outputVariable)
            dataset.append(datapoint)
            i = i + 1
        dataset = pd.DataFrame(dataset)
        dataset = self.addNoise(dataset, noiseLevel)
        return dataset

    def __init__(self, knownFunction, numberOfDatapoints, noiseLevel, inputVariables,  outputVariable):
        self.knownFunction = knownFunction
        self.numberOfDatapoints = numberOfDatapoints
        self.noiseLevel = noiseLevel
        self.outputVariable = outputVariable
        self.inputVariables = inputVariables
        self.Dataset = self.createDataset(self.knownFunction, self.numberOfDatapoints, self.noiseLevel,  self.inputVariables, self.outputVariable)

class RelativeDifferenceLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(RelativeDifferenceLoss, self).__init__(name='RelativeDifferencePercentage')

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        lossValue = (tf.math.abs(y_pred - y_true) / tf.math.maximum(tf.math.abs(y_pred), tf.math.abs(y_true)))
        return tf.reduce_mean(lossValue*100, axis = -1)

class MyModel(tf.keras.Model):
    def __init__(self,layerTopology, summaryWriter, piMatrix = None):
        super(MyModel,self).__init__()
        self.piMatrix = piMatrix
        self.current_epoch = tf.Variable(tf.random.uniform((1,1)), shape=(1,1))
        self.summaryWriter = summaryWriter
        self.layerTopology = layerTopology
        self.createdLayers = []
        for j in range(2,len(self.layerTopology.index) -1 ):
            if self.layerTopology['Type'][j] == 'functionalBlock':
                newLayer = MyFunctionBlock(self.layerTopology['Name'][j])
                self.createdLayers.append(newLayer)
        self.piLayer = MyProductBlock('piLayer',[self.piMatrix.shape[0]-1])
        self.shortcutLayer = MyProductBlock('shortcutLayer')
        self.finalLayer = keras.layers.Dense(1, name ='FinalLayer', use_bias = False)

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
        tf.print(self.current_epoch)
        tf.print(gradients, summarize=-1)
        tf.print(gradients[1][5], summarize=-1)
        with self.summaryWriter.as_default():
            tf.summary.histogram("X-Data", x,step =  self.current_epoch.numpy())
            tf.summary.histogram("Y-Data", y,step =  self.current_epoch.numpy())
            tf.summary.scalar('LNGrad',tf.reshape(gradients[1][5],[]),step =  self.current_epoch.numpy())
            tf.summary.scalar('LNGrad1',tf.reshape(gradients[3][5],[]),step =  self.current_epoch.numpy())
            tf.summary.scalar('LNWeight',tf.reshape(trainable_vars[1][5],[]),step =  self.current_epoch.numpy())
            tf.summary.scalar('LNWeight1',tf.reshape(trainable_vars[3][5],[]),step =  self.current_epoch.numpy())
            for gradient in gradients:
                tf.summary.histogram(gradient.name, gradient,step = self.current_epoch.numpy())
            for var in trainable_vars:
                tf.summary.histogram("Gewichte", var,step = self.current_epoch.numpy())
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}       

    def model_graph(self):
        x = keras.Input(shape=(self.piMatrix.shape[-1] -1))
        return keras.Model(inputs=[x], outputs=self.call(x))

    def _getWeights(self):
        self.piLayerWeight = np.transpose(self.piMatrix.values[1:,1:])
        self.shortCutLayerWeight = (self.piMatrix.values[0,1:]).reshape(-1, 1) * (-1)
    
    def _setWeights(self):
        self.piLayer.set_weights([self.piLayerWeight])
        self.piLayer.trainable = False
        self.shortcutLayer.set_weights([self.shortCutLayerWeight])
        self.shortcutLayer.trainable = False

class MyBlock(tf.keras.layers.Layer):
    def createMoreLayers(self, internalTopology, blockName, initializer):
        layerList = []
        for i in range(0,len(internalTopology)):
            layerList.extend([keras.layers.Dense(internalTopology[i],use_bias = False, name = (blockName + '_Layer_'+ str(i)),kernel_initializer=initializer)])
        return layerList

    def __init__(self, blockName, internalTopology, initializer = 'random_uniform'):
        super(MyBlock, self).__init__()
        self.initializer = initializer
        self.blockName = blockName
        self.internalTopology = internalTopology
        self.additionalLayers = self.createMoreLayers(self.internalTopology, self.blockName, self.initializer)

    def call(self, input_tensor):
        x = input_tensor
        for Layer in self.additionalLayers:
            x = Layer(x)
        return x

class MyModelInterpretation():
    def __init__(self, model):
        self.model = model
        self.formula = ''
        self.InputVariableNames = self.model.piMatrix.columns[1:].tolist()
    def _getInputLayers(self):
        inputLayers = []
        for layer in self.model.layers:
            inboundNodesNames = [inboundNode.inbound_layers.__class__.__name__ for inboundNode in layer.inbound_nodes]
            if ('InputLayer' in inboundNodesNames):
                inputLayers.append(layer)
        return inputLayers
    
    def _getLayerFormula(self, Layer, inputVariables):
        # get Type of Layer
        layerType = Layer.__class__.__name__
        ## get LayerFromula
        switcher={
            'MyFunctionBlock': self._internalFormulaFunctionBlock,
            'Dense': self._internalFormulaDenseLinearLayer,
            'Concatenate':self._internalFormulaConcatenateLayer,
            'MyProductBlock':self._internalFormulaProductBlock,
            'Multiply':self._internalFormulaMultiplyLayer
        }
        getLayerFormulaFunction = switcher.get(layerType)
        LayerFormula = getLayerFormulaFunction(Layer, inputVariables)
        # update Layer Formulas
        self.layerFormulas[Layer.name]=LayerFormula
        #print(Layer.name)
        # search for next Layer if LayerFormula is not None
        if LayerFormula is not None:
            for i in range(0,int(math.ceil(len(Layer.outbound_nodes)/2))):
                nextLayer = Layer.outbound_nodes[i].outbound_layer
                self._getLayerFormula(nextLayer, LayerFormula)
    
    def _internalFormulaFunctionBlock(self, layer, inputVariables):
        LayerFormula = []
        # inputs inkl fn_map
        fn_maps = ['x','sin(x)','cos(x)','tanh(x)','e^(x)','log(x)','1/(1+e^(x))']
        formulaAfterMaps = []
        for fn_map in fn_maps:
            for inputVariable in inputVariables:
                formulaAfterMaps.append(fn_map.replace('x',inputVariable))
        ## Product-Block
        # internal Weights
        productWeights = np.eye((layer.ProductBlock.weights[0]).shape[0])
        for weights in layer.ProductBlock.weights:
            productWeights = np.matmul(productWeights, weights.numpy())
        productWeights = productWeights.T.tolist()[0]
        for i in range(0,len(productWeights)):
            ##if productWeights[i] < 0.01:
            ##    productWeights[i] = 0
            productWeights[i] = Fraction(round(productWeights[i],3)).limit_denominator(1000)
        #Combine Weights with inputs after mapping
        productFormula = []
        for i in range(0,len(formulaAfterMaps)):
            if productWeights[i].numerator != 0:
                productFormula.append('('+str(productWeights[i])+')' + '*' +'log(' + str(formulaAfterMaps[i]) + ')')
        seperator = '+'
        productFormula = 'e^(' + (seperator.join(productFormula)) + ')'

        ## Sum-Block
        # internal Weights
        sumWeights = np.eye((layer.SumBlock.weights[0]).shape[0])
        for weights in layer.SumBlock.weights:
            sumWeights = np.matmul(sumWeights, weights.numpy())
        sumWeights = sumWeights.T.tolist()[0]
        for i in range(0,len(sumWeights)):
            ##if sumWeights[i] < 0.01:
            ##    sumWeights[i] = 0
            sumWeights[i] = Fraction(round(sumWeights[i],3)).limit_denominator(1000)
        #Combine Weights with inputs after mapping
        sumFormula = []
        for i in range(0,len(formulaAfterMaps)):
            if sumWeights[i].numerator != 0:
                sumFormula.append('(' + str(sumWeights[i]) + ')' + '*' +str(formulaAfterMaps[i]))
        seperator = '+'
        sumFormula = seperator.join(sumFormula)
        LayerFormula = [sumFormula, productFormula]
        return LayerFormula

    def _internalFormulaDenseLinearLayer(self, layer, inputVariables):
        LayerFormula = []
        denseWeights = layer.weights[0].numpy().T.tolist()[0]
        for i in range(0,len(denseWeights)):
            ##if denseWeights[i] < 0.01:
            ##    denseWeights[i] = 0
            denseWeights[i] = Fraction(round(denseWeights[i],3)).limit_denominator(1000)
        for i in range(0,len(inputVariables)):
            if denseWeights[i].numerator != 0:
                LayerFormula.append('(' + str(denseWeights[i]) + ')' + '*' + inputVariables[i])
        seperator = '+'
        LayerFormula = seperator.join(LayerFormula)
        return LayerFormula

    def _internalFormulaConcatenateLayer(self, layer, inputVariables):
        LayerFormula = []
        #get inputs from linked layers
        for inboundlayer in layer.inbound_nodes[0].inbound_layers:
            if self.layerFormulas[inboundlayer.name] is not None:
                LayerFormula = LayerFormula + self.layerFormulas[inboundlayer.name]

        #check if all input layers are ready if not ->
        if len(LayerFormula) != layer.output_shape[1]:
            LayerFormula = None

        return LayerFormula
    
    def _internalFormulaProductBlock(self, layer, inputVariables):
        ## Product-Block
        # internal Weights
        layerFormula = []
        productWeights = np.eye((layer.weights[0]).shape[0])
        for weights in layer.weights:
            productWeights = np.matmul(productWeights, weights.numpy())
        for j in range(0,productWeights.shape[-1]):
            productWeights_temp = productWeights.T.tolist()[j]
            for i in range(0,len(productWeights_temp)):
                ##if productWeights[i] < 0.01:
                ##    productWeights[i] = 0
                productWeights_temp[i] = Fraction(round(productWeights_temp[i],3)).limit_denominator(1000)
            productFormula = []
            #Combine Weights with inputs after mapping
            for i in range(0,len(inputVariables)):
                if productWeights_temp[i].numerator != 0:
                    productFormula.append('(' + str(productWeights_temp[i]) + ')' +'*' +'log(' + str(inputVariables[i]) + ')')
            seperator = '+'
            productFormula = 'e^(' + (seperator.join(productFormula)) + ')'
            layerFormula.append(productFormula)
        return layerFormula

    def _internalFormulaMultiplyLayer(self, layer, inputVariables):
        LayerFormula_temp = []
        LayerFormula = []
        #get inputs from linked layers
        for inboundlayer in layer.inbound_nodes[0].inbound_layers:
            if self.layerFormulas[inboundlayer.name] is not None:
                LayerFormula_temp = LayerFormula_temp + self.layerFormulas[inboundlayer.name]

        #check if all input layers are ready if not return None
        totalInputs = 0
        for inputShape in layer.input_shape:
            totalInputs = totalInputs + inputShape[1]

        if len(LayerFormula_temp) != totalInputs:
            LayerFormula = None
        else:
            if layer.input_shape[0][1] == 1:
                onedimensionalInput = 0
            else:
                 onedimensionalInput = layer.input_shape[0][1]
            i = 0
            for inputFormula in LayerFormula_temp:
                if i != onedimensionalInput:
                    LayerFormula = LayerFormula + ([inputFormula + '*' +LayerFormula_temp[onedimensionalInput]])
                i = i +1

        return LayerFormula
    
    def get_formula(self):
        self.model.model_graph()
        ## reset Layerformulas
        self.layerFormulas = {}
        for layer in self.model.layers:
            self.layerFormulas[layer.name] = None
        ##get Model Input
        self.modelInputName = self.model.input_names
        ## get InputLayers
        self.inputLayers = self._getInputLayers()
        ## start the recursive formula search from all input layers
        for inputLayer in self.inputLayers:
            self._getLayerFormula(inputLayer,self.InputVariableNames)

class MyFunctionBlock(tf.keras.layers.Layer):
    def __init__(self, blockname, internalTopologyProduct = [1], internalTopologySum= [1],  applicableFunctions=[tf.math.sin, tf.math.cos]):
        super(MyFunctionBlock, self).__init__()
        self.applicableFunctions = applicableFunctions
        self.internalTopologyProduct = internalTopologyProduct
        self.internalTopologySum = internalTopologySum
        self.blockname = blockname
        ###### 
        self.SumBlock = MyBlock(self.blockname + '_Sum', self.internalTopologySum)
        self.ProductBlock  = MyProductBlock(self.blockname + '_Product', self.internalTopologyProduct)

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
    def __init__(self, blockname, internalTopologySum = [1], initializer = 'random_normal'):
        super(MyProductBlock, self).__init__()
        self.internalTopologySum = internalTopologySum
        self.blockname = blockname
        self.SumBlock = MyBlock(self.blockname + '_Sum', self.internalTopologySum, initializer)

    def call(self,input_tensor):
        x = input_tensor
        x = tf.map_fn(tf.math.abs, x)
        x = tf.map_fn(tf.math.log, x)
        x = self.SumBlock(x)
        x = tf.map_fn(tf.math.exp, x)
        return x

class MyCallback(keras.callbacks.Callback):
    def __init__(self, log_dir, matlabEng, standardDeviation = 0.01):
        self.log_dir = log_dir
        self.matlabEng = matlabEng
        self.standardDeviation = standardDeviation

    def on_epoch_begin(self, epoch, logs=None):
        self.model.current_epoch = self.model.current_epoch + 1
            
    def on_epoch_end(self, epoch, logs=None):
        with open((self.log_dir + '/weights.txt'), "a") as myWeightfile:
        ## Save weights
            print(('Epoch '+ str(epoch)), file=  myWeightfile)
            print(self.model.weights.__str__(), file= myWeightfile )
        with open((self.log_dir + '/Formula.txt'), "a") as myFormulafile:
        ## Save formula
            print('      Formel:', file= myFormulafile )
            formula = MyModelInterpretation(self.model)
            formula.get_formula()
            formulastring = formula.layerFormulas['FinalLayer']
            print(formulastring, file= myFormulafile)
        with open((self.log_dir + '/FormulaLatex.txt'), "a") as myFormulaLatexfile:
        ## Save Latex
            formulastring_matlab = formulastring.replace("e^", "exp")
            formulastring_latex = self.matlabEng.Simplifier(formulastring_matlab, (list(self.model.piMatrix.columns[1:])))
            print(formulastring_latex, file= myFormulaLatexfile)
        ## Early stopping
        numberOfInputs = self.model._build_input_shape[-1]
        if(len(self.model.history.epoch) > 5 and logs['loss'] < max(numberOfInputs * self.standardDeviation * 100, 0.1)):
            for j in range(1,6):
                loss1 = self.model.history.history['loss'][-j] 
                loss2 = self.model.history.history['loss'][-j-1]
                if np.abs(loss1 - loss2) > 0.01:
                    self.model.stop_training = False
                    break
                else: 
                    self.model.stop_training = True
        print(logs.keys())

    def on_batch_begin(self, batch, logs=None):
        self.model.current_epoch = self.model.current_epoch + 1
        tf.print(self.model.current_epoch)
    
class MsTeamsMessenger():
    def __init__(self):
        import pymsteams
        self.myTeamsMessage = pymsteams.connectorcard('https://outlook.office.com/webhook/4fa26457-a51f-423b-9570-f0dc67db190b@b6ed300d-a38f-4a80-8c3d-192e250c0f65/IncomingWebhook/0efc61a8c33d48628e4046b892bf241f/adef8b1f-6d30-4b20-a8fd-50890efc4196')
    
    def __call__(self,message):
        self.myTeamsMessage.text(message)
        self.myTeamsMessage.send()

class ImportConfig():
    print('a')

class MatlabAdapter():
    print('a')
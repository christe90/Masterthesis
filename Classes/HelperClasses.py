import datetime
import json
import pandas as pd
import numpy as np
import math
import DHNNClasses as dhnn
import tensorflow as tf
import realFunctions as rf
import matlab.engine
from fractions import Fraction
from tensorflow import keras

class Experiment():
    def __init__(self,configFile):
        ## Pfad zum ConfigFile speichern
        self.configFile = configFile
        ## Startzeitpunkt als String speichern um diesen für die Log-Ordnernamen zu verwenden
        self.startTime = (datetime.datetime.now()).strftime("%Y%m%d%H%M%S")
    
    def get_config(self):
        with open(self.configFile) as json_file:
            self.config = json.load(json_file)
    
    def get_Data(self):
        ## Trainingsdaten einlesen
        with open(self.config['path_data_file']) as data_file:
            self.data = pd.read_csv((data_file),sep=';')
        ## Topologietabelle einlesen
        with open(self.config['path_topology_file']) as topology_file:
            self.topology = pd.read_csv((topology_file),sep=';')
        ## Dimensionsmatrix einlesen
        with open(self.config['path_dimensionsmatrix_file']) as dimensionsmatrix_file:
            self.dimensionsmatrix = pd.read_csv((dimensionsmatrix_file),sep=';',index_col = 0)
    
    def get_Dataset(self):
        self.dataset = dhnn.Dataset(dimensionalMatrix= self.dimensionsmatrix,
                    data = self.data, 
                    splitRatio= self.config['split_ratio'], 
                    y_Name= self.config['y_name'],
                    augmentationFactor = self.config['data_augmentation_factor'])
    
    def create_Model(self):
        self.summaryWriter = tf.summary.create_file_writer(self.config['path_experiment'] +"/mylogs/eager/"+ self.startTime+'/')
        self.model  = dhnn.MyModel(layerTopology = self.topology,
                                    summaryWriter = self.summaryWriter,
                                    piMatrix = self.dataset.piMatrix,
                                    dimensionless = self.config['dimensionhomogenous'])
    
    def compile_Model(self):
        self.model.compile(
                loss= self.config['loss_function'],
                optimizer = self.config['optimizer'],
                metrics = self.config['metrics'], 
                )

    def plot_Model(self):
        tf.keras.utils.plot_model(
                self.model.model_graph(),                   
                to_file=(self.config['path_experiment']  + '/modelplot.png'),
                dpi=96,            
                show_shapes=True, 
                show_layer_names=True,
                expand_nested=True, 
                rankdir='LR'                     
                )
    
    def train_Model(self):
        ## initialize Callbacks
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                                        log_dir=(self.config['path_experiment'] + '/logs/'+ self.startTime +'/'),
                                                         histogram_freq=1)
        self.log_level = self.config['log_level']
        self.model.set_log_level(self.log_level, self.config['path_experiment'])
        myCallbacks = MyCallback(self.config['path_experiment'], self.log_level, self.config['early_stopping_limit'])
        ##Training starten
        self.model.fit(x=self.dataset.trainingData['x_Values'], 
                    y=self.dataset.trainingData['y_Values'], 
                    epochs=self.config['max_epochs'],
                    batch_size = self.config['batch_size'],
                    validation_data=(self.dataset.testData['x_Values'], self.dataset.testData['y_Values']),
                    callbacks = [tensorboard_callback, myCallbacks])

    def save_Model(self):
        self.model.save(self.config['path_experiment']+ "/" + self.startTime + "_Model")
    
    def interpret_Model(self):
        if self.config['path_dc43_file'] == '':
            path_dc43_file = None
        else:
            path_dc43_file = self.config['path_dc43_file']
        if self.config['path_latex_file'] == '':
            path_latex_file = None
        else:
            path_latex_file = self.config['path_latex_file']
        variableNames = (self.config['x_names']) + list(self.config['y_name'])
        interpreter = MyModelInterpretation(model = self.model, path43 = path_dc43_file, pathLatex = path_latex_file, variableNames = variableNames)
        interpreter.getFormula()

class MyCallback(keras.callbacks.Callback):
    def __init__(self, log_dir,log_level, early_stopping_limit):
        self.log_dir = log_dir
        self.log_level = log_level
        self.early_stopping_limit = early_stopping_limit

    def on_epoch_begin(self, epoch, logs=None):
        if self.log_level == 3:
            ## Save weights
            with open((self.log_dir + '/weights.txt'), "a") as myWeightfile:
                print(('Epoch '+ str(epoch)), file=  myWeightfile)
                try:
                    print(self.model.weights.__str__(), file= myWeightfile )
                except:
                    print('Model hat keine Gewichte', file= myWeightfile )
            ## Gradienttrenner einfügen
            with open((self.log_dir + '/gradients.txt'), "a") as myGradientfile:
                print(('Epoch '+ str(epoch)), file=  myGradientfile)

    def on_epoch_end(self, epoch, logs=None):
        ## Early stopping
        if(len(self.model.history.epoch) > 5 and logs['loss'] < self.early_stopping_limit):
            for j in range(1,6):
                loss1 = self.model.history.history['loss'][-j] 
                if loss1 > self.early_stopping_limit:
                    self.model.stop_training = False
                    break
                else: 
                    self.model.stop_training = True

    def on_batch_end(self, batch, logs=None):
        if self.log_level == 3:
            ## Save weights
            with open((self.log_dir + '/weights.txt'), "a") as myWeightfile:
                print(('Batch end: '+ str(batch)), file=  myWeightfile)
                print(self.model.weights.__str__(), file= myWeightfile )
            ## Gradienttrenner einfügen
            with open((self.log_dir + '/gradients.txt'), "a") as myGradientfile:
                print(('Batch '+ str(batch)), file=  myGradientfile)

    def on_batch_begin(self, batch, logs=None):
        if self.log_level == 3:
            ## Save weights
            with open((self.log_dir + '/weights.txt'), "a") as myWeightfile:
                print(('Batch '+ str(batch)), file=  myWeightfile)
                try:
                    print(self.model.weights.__str__(), file= myWeightfile )
                except:
                    print('Model hat keine Gewichte', file= myWeightfile )
            ## Gradienttrenner einfügen
            with open((self.log_dir + '/gradients.txt'), "a") as myGradientfile:
                print(('Batch '+ str(batch)), file=  myGradientfile)

class FakeDataGenerator():
    def __init__(self, x_Names, y_Name, realFunctionName):
        self.x_Names = x_Names
        self.y_Name = y_Name
        self.realFunctionName = realFunctionName
        self.myFunction = rf.realFunctions(self.realFunctionName)
    def createDatapoints(self, numberOfDatapoints):
        datapoints = []
        for i in range(0,numberOfDatapoints):
            datapoint = self._createDataPoint()
            datapoints.append(datapoint)
        self.datapoints = pd.DataFrame(datapoints)

    def addNoise(self, noiseLevel):
        noiseSet = pd.DataFrame(1 + np.random.normal(0, noiseLevel, self.datapoints.shape),columns = self.datapoints.columns)
        self.datapoints = self.datapoints * noiseSet

    def _createDataPoint(self):
        dataPoint = {}
        for variable in self.x_Names:
            dataPoint[variable] = np.random.randint(1,10)
        dataPoint[self.y_Name] = self.myFunction.realFunction(**dataPoint)
        return dataPoint

class MyModelInterpretation():
    def __init__(self, model, path43, pathLatex, variableNames):
        self.model = model
        self.path43 = path43
        self.pathLatex = pathLatex
        self.formula = ''
        self.InputVariableNames = self.model.piMatrix.columns[1:].tolist()
        self.model.model_graph()
        self.variableNames = variableNames
        self.layerFormulas = {}
        self.submoduleNames = self._getSubmodules()
    
    def _getSubmodules(self):
        modulenames = []
        for layer in self.model.layers:
            for submodul in layer.submodules:
                modulenames.append(submodul.name)
        return modulenames

    def getFormula(self):
        ##alle InputLayer ermitteln
        self.inputLayers = self._getInputLayers()
        ## von den InputLayern ausgehend die Die Formeln der folgenden Layern ermitteln
        for inputLayer in self.inputLayers:
            self._getLayerFormula(inputLayer,self.InputVariableNames)
        ##Ausgabe des Final Layers
        self.formula = self.layerFormulas['FinalLayer']
        ## DC43 Export:
        if self.path43 is not None:
            with open(self.path43, "a") as file43:
                file43.write(self.formula)
        ## Latex Export:
        ##if self.pathLatex is not None:
        ##    matlabEng = MatlabAdapter()
        ##    formulaLatex = matlabEng(self.formula, self.variableNames)
        ##    with open(self.pathLatex, "a") as fileLatex:
        ##        fileLatex.write(formulaLatex)

    def _getInputLayers(self):
        ## Es werden Layer gesucht die mit dem Inputlayer verbunden sind
        inputLayers = []
        for layer in self.model.layers:
            inboundNodesNames = [inboundNode.inbound_layers.__class__.__name__ for inboundNode in layer.inbound_nodes]
            if ('InputLayer' in inboundNodesNames):
                inputLayers.append(layer)
        return inputLayers
    
    def _getLayerFormula(self, Layer, inputVariables):
        # get Type of Layer
        layerType = Layer.__class__.__name__
        ## get LayerFromulaType
        switcher={
            'MyFunctionBlockFull': self._internalFormulaFunctionBlockFull,
            'MyFunctionBlock': self._internalFormulaFunctionBlock,
            'Dense': self._internalFormulaDenseLinearLayer,
            'Concatenate':self._internalFormulaConcatenateLayer,
            'MyProductBlock':self._internalFormulaProductBlock,
            'Multiply':self._internalFormulaMultiplyLayer
        }
        ## Get the Formula of the current Layer from the Function from Switcher
        getLayerFormulaFunction = switcher.get(layerType)
        LayerFormula = getLayerFormulaFunction(Layer, inputVariables)
        # update Formula of current Layer
        if LayerFormula is not None:
            self.layerFormulas[Layer.name]=LayerFormula
        # search for next Layer if LayerFormula is not None
        if LayerFormula is not None:
            for i in range(0,int(math.ceil(len(Layer.outbound_nodes)/2))):
                nextLayer = Layer.outbound_nodes[i].outbound_layer
                if nextLayer.name not in self.submoduleNames:
                    self._getLayerFormula(nextLayer, LayerFormula)
    
    def _getProductFormula(self, formulaAfterMaps, layerWeights):
        productFormulaLayer = []
        ## Gewichte in eine Liste schreiben
        layerWeights = layerWeights.numpy().T.tolist()
        for weights in layerWeights:
            productFormula = []
            ## Gewichte zu Brüchen mit max 1000 im Nenner machen
            for i in range(0,len(weights)):
                if abs(weights[i]) < 0.01:
                     weights[i] = 0
                weights[i] = Fraction(round(weights[i],3)).limit_denominator(1000)
            ## Formel aus den Gewichten und Inputs machen
            for i in range(0,len(formulaAfterMaps)):
                if weights[i].numerator != 0:
                    productFormula.append('('+str(weights[i])+')' + '*' +'log(' + str(formulaAfterMaps[i]) + ')')
            ## Zusammensetzen der einzelnen Formeln
            seperator = '+'
            productFormula = '(e^(' + (seperator.join(productFormula)) + '))'
            productFormulaLayer.append(productFormula)
        return productFormulaLayer

    def _getSumFormula(self, formulaAfterMaps, weights, bias = None):
        sumFormula = []
        ## Gewichte in eine Liste schreiben
        weights = weights.numpy().T.tolist()[0]
        if bias is not None:
            bias = bias.numpy()[0]
            if abs(bias) < 0.01:
                bias = 0
            bias = Fraction(str(round(bias,3))).limit_denominator(1000)
        ## Gewichte zu Brüchen mit max 1000 im Nenner machen
        for i in range(0,len(weights)):
            if abs(weights[i]) < 0.01:
                weights[i] = 0
            weights[i] = Fraction(round(weights[i],3)).limit_denominator(1000)
        ## Formel aus den Gewichten und Inputs machen
        for i in range(0,len(formulaAfterMaps)):
            if weights[i].numerator != 0:
                sumFormula.append('(' + str(weights[i]) + ')' + '*' +str(formulaAfterMaps[i]))
        seperator = '+'
        sumFormula = seperator.join(sumFormula)
        if bias is None or bias.numerator == 0:
            return '(' +sumFormula + ')'
        else:
            return '(' + sumFormula + '+(' + str(bias) + '))'

    def _get_fn_maps(self, layer):
        if layer.fn_type == 'sin':
            fn_maps = ['x','sin(x)']
        elif layer.fn_type == 'cos':
            fn_maps = ['x','cos(x)']
        elif layer.fn_type == 'tanh':
            fn_maps = ['x','tanh(x)']
        elif layer.fn_type == 'exp':
            fn_maps = ['x','e^(x)']
        elif layer.fn_type == 'log':
            fn_maps = ['x','log(x)']
        elif layer.fn_type == 'sig':
            fn_maps = ['x','1/(1+e^(x))']
        elif layer.fn_type == 'lin':
            fn_maps = ['x']

        return fn_maps

    def _internalFormulaFunctionBlockFull(self, layer, inputVariables):
        # inputs inkl fn_map
        fn_maps = ['x','sin(x)','cos(x)','tanh(x)','e^(x)','log(x)','1/(1+e^(x))']
        formulaAfterMaps = []
        for fn_map in fn_maps:
            for inputVariable in inputVariables:
                formulaAfterMaps.append(fn_map.replace('x',inputVariable))
        ## Product-Block
        productFormula = self._getProductFormula(formulaAfterMaps, layer)
        sumFormula = self._getSumFormula(formulaAfterMaps, layer)
        return [sumFormula, productFormula]

    def _internalFormulaFunctionBlock(self, layer, inputVariables):
        # inputs inkl fn_map
        fn_maps = self._get_fn_maps(layer)
        formulaAfterMaps = []
        for fn_map in fn_maps:
            for inputVariable in inputVariables:
                formulaAfterMaps.append(fn_map.replace('x',inputVariable))
        ## Product-Block
        productFormula = self._getProductFormula(formulaAfterMaps, layer.weights[0])
        ## SumBlock
        sumFormula = self._getSumFormula(formulaAfterMaps, layer.weights[1], layer.weights[2])
        return [productFormula[0], sumFormula]

    def _internalFormulaProductBlock(self, layer, inputVariables):
        productFormula = self._getProductFormula(inputVariables, layer.weights[0])
        return productFormula

    def _internalFormulaDenseLinearLayer(self, layer, inputVariables):
        sumFormula = self._getSumFormula(inputVariables, layer.weights[0])
        return sumFormula

    def _internalFormulaConcatenateLayer(self, layer, inputVariables):
        LayerFormula_temp = []
        LayerFormula = []
        #get inputs from linked layers
        for inboundlayer in layer.inbound_nodes[0].inbound_layers:
            if inboundlayer.name in self.layerFormulas.keys():
                LayerFormula_temp = LayerFormula_temp + self.layerFormulas[inboundlayer.name]

        #check if all input layers are ready if not ->
        if len(LayerFormula_temp) != layer.output_shape[1]:
            LayerFormula = None
        else:
            LayerFormula = LayerFormula_temp

        return LayerFormula
    
    def _internalFormulaMultiplyLayer(self, layer, inputVariables):
        LayerFormula_temp = []
        LayerFormula = []
        #get inputs from linked layers
        for inboundlayer in layer.inbound_nodes[0].inbound_layers:
            if inboundlayer.name in self.layerFormulas.keys():
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

class MsTeamsMessenger():
    def __init__(self):
        import pymsteams
        self.messages = {'Success':'Der Versuch {} ist mit Loss: {} durchgelaufen',
                        'Error':'Der Versuch {} ist mit einem Fehler abgebrochen'}
        with open('C:/Users/Uni/Documents/msTeamsAPI.txt') as api_file:
            self.apiKey = api_file.read()
        self.myTeamsMessage = pymsteams.connectorcard(self.apiKey)
    
    def __call__(self,messageType, ExperimentName, Loss = 0):
        messageText = self.messages[messageType]
        messageText = messageText.format(ExperimentName, str(Loss))
        self.myTeamsMessage.text(messageText)
        self.myTeamsMessage.send()

class MatlabAdapter():
    def __init__(self):
        self.engine = matlab.engine.start_matlab()
    def __call__(self, formulaString, variableNames):
        formulastring_matlab = formulaString.replace("e^", "exp")
        formulastring_latex = self.engine.Simplifier(formulastring_matlab, variableNames)
        return formulastring_latex
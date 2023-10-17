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

class MsTeamsMessenger():
    def __init__(self):
        import pymsteams
        self.myTeamsMessage = pymsteams.connectorcard('https://outlook.office.com/webhook/4fa26457-a51f-423b-9570-f0dc67db190b@b6ed300d-a38f-4a80-8c3d-192e250c0f65/IncomingWebhook/0efc61a8c33d48628e4046b892bf241f/adef8b1f-6d30-4b20-a8fd-50890efc4196')
    
    def __call__(self,message):
        self.myTeamsMessage.text(message)
        self.myTeamsMessage.send()

class Configuration():
    print('a')

class MatlabAdapter():
    print('a')

class Experiment():
    print('a')
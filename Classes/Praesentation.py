import DHNNClasses as dhnn
import HelperClasses as hc

def modifyNumberDatapoint(referenceSize):
    import json
    filepaths = ['/Topo/ConfigFile.json',
            '/NoDH/ConfigFile.json',
            '/Aug10/ConfigFile.json']
    for filepath in filepaths:
        with open(filepath, 'r') as file:
            json_object = json.load(file)
        with open(filepath, 'w') as file:
            json_object["number_datapoints"] = referenceSize
            json_object["max_epochs"] = 5000
            json.dump(json_object, file)
    

config_File = '/ConfigFile.json'
for i in range(0,9):
    check = 0
    for counter in range(0,5):
        with open('/currentI.txt', 'w') as f:
            print('i: ' + str(i), file=f)
        myExperiment = hc.Experiment(config_File)
        myMessenger = hc.MsTeamsMessenger()
        myExperiment.get_config()
        myExperiment.config['number_datapoints'] = myExperiment.config['number_datapoints'] * 2**i
        myExperiment.get_Data()
        if myExperiment.config['number_datapoints'] != 0:
            dataGen = hc.FakeDataGenerator(x_Names = myExperiment.config['x_names'], y_Name = myExperiment.config['y_name'], realFunctionName = myExperiment.config['name_experiment'])
            dataGen.createDatapoints(myExperiment.config['number_datapoints'])
            myExperiment.data = dataGen.datapoints
        myExperiment.get_Dataset()
        myExperiment.create_Model()
        myExperiment.compile_Model()
        myExperiment.model.model_graph()
        myExperiment.plot_Model()
        try:
            myExperiment.train_Model()
            myMessenger('Success',myExperiment.config['name_experiment'], myExperiment.model.history.history['loss'][-1])
            myExperiment.save_Model()
            myExperiment.interpret_Model()
            if myExperiment.model.history.history['loss'][-1] < 20:
                modifyNumberDatapoint(myExperiment.config['number_datapoints'])
                check = check + 1
            if check > 1:
                break
        except:
            if myExperiment.model.history.history['loss'][-1] < 20:
                modifyNumberDatapoint(myExperiment.config['number_datapoints'])
                check = check + 1
            if check > 1:
                break
    if check > 1:
        break
    
config_Files = ['/Topo/ConfigFile.json',
                '/NoDH/ConfigFile.json',
                '/Aug10/ConfigFile.json']
for config_File in config_Files:
    for counter in range(0,7):
        myExperiment = hc.Experiment(config_File)
        myMessenger = hc.MsTeamsMessenger()
        myExperiment.get_config()
        myExperiment.config['number_datapoints'] = myExperiment.config['number_datapoints']
        myExperiment.get_Data()
        if myExperiment.config['number_datapoints'] != 0:
            dataGen = hc.FakeDataGenerator(x_Names = myExperiment.config['x_names'], y_Name = myExperiment.config['y_name'], realFunctionName = myExperiment.config['name_experiment'])
            dataGen.createDatapoints(myExperiment.config['number_datapoints'])
            myExperiment.data = dataGen.datapoints
        myExperiment.get_Dataset()
        myExperiment.create_Model()
        myExperiment.compile_Model()
        myExperiment.model.model_graph()
        myExperiment.plot_Model()
        try:
            myExperiment.train_Model()
            myMessenger('Success',myExperiment.config['name_experiment'], myExperiment.model.history.history['loss'][-1])
            myExperiment.save_Model()
            myExperiment.interpret_Model()
        except:
           print('FEHLER')

import DHNNClasses as dhnn
import HelperClasses as hc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
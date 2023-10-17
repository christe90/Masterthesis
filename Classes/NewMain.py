import DHNNClasses as dhnn
import HelperClasses as hc

config_File = 'ConfigFile.json'
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
myExperiment.train_Model()
myExperiment.save_Model()
myExperiment.interpret_Model()

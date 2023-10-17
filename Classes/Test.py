import DHNNClasses as dhnn
import HelperClasses as hc
i = 5
check = 0
config_File = 'C/ConfigFile.json'
for counter in range(0,1):
    with open('C/currentI.txt', 'w') as f:
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
    ##try:
    myExperiment.train_Model()
    myMessenger('Success',myExperiment.config['name_experiment'], myExperiment.model.history.history['loss'][-1])
    myExperiment.save_Model()
    myExperiment.interpret_Model()
    if myExperiment.model.history.history['loss'][-1] < 20:
        modifyNumberDatapoint(myExperiment.config['number_datapoints'])
        check = check + 1
    if check > 1:
        break
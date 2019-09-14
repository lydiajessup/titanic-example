const TRAIN_DATA_PATH = 'data/titanic_train.csv';
const TEST_DATA_PATH = 'data/titanic_test.csv';


let neuralNet;
const dataset_options = {columnConfigs: {survived: {isLabel: true}}}

async function setup() {

    neuralNet = new NeuralNet();
   
    await neuralNet.loadCSV(TRAIN_DATA_PATH,  'training', dataset_options);
    await neuralNet.loadCSV(TEST_DATA_PATH,  'testing', dataset_options);
    await neuralNet.loadCSV(TRAIN_DATA_PATH,  'validating', dataset_options);
    await neuralNet.prepareData('training');
    await neuralNet.prepareData('testing');
    await neuralNet.prepareData('validating');

    await neuralNet.train();
    // console.log(neuralNet);
    
  }

setup();
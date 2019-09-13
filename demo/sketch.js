const TRAIN_DATA_PATH = 'data/titanic_train.csv';
const TEST_DATA_PATH = 'data/titanic_test.csv';


let neuralNet;
const dataset_options = {columnConfigs: {survived: {isLabel: true}}}

async function setup() {

    neuralNet = new NeuralNet();
   
    await neuralNet.loadCSV(TRAIN_DATA_PATH, dataset_options);
    await neuralNet.summarize('training');
    // console.log(neuralNet);
    
  }

setup();
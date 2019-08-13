// Title:       titanic suvival prediction
// Date:        August 13, 2019
// Created by:  Lydia Jessup
//              Using an example from TensorFlow.js here:
//              https://www.tensorflow.org/js/tutorials/training/nodejs_training
// Description: This example uses pre-cleaned data to train a neural network
//              to predict survival on the titanic

//load tensorflow
const tf = require('@tensorflow/tfjs');

///////////////////////////////////////////////
// Import, normalize and transform data
///////////////////////////////////////////////

// util function to normalize a value between a given range.
function normalize(value, min, max) {
  if (min === undefined || max === undefined) {
    return value;
  }
  return (value - min) / (max - min);
}

// data can be loaded from URLs or local file paths when running in Node.js.
// in this case loading pre-cleaned and split data
const TRAIN_DATA_PATH = 'file://titanic_train.csv';
const TEST_DATA_PATH = 'file://titanic_test.csv';


//// Make Constants from training data ////

// have to put in all min and max values in order to normalize
//age and fare -- for class and sex will have to do one hot encoding
const AGE_MIN = 0.0;
const AGE_MAX = 80.0;
const FARE_MIN = 0.0;
const FARE_MAX = 512.0;

const NUM_SURVIVED_CLASSES = 2;
const TRAINING_DATA_LENGTH = 916;
const TEST_DATA_LENGTH = 393;


//// Convert rows from the CSV into features and labels ////

// Each feature field is normalized within training data constants
// xs is the features and xy is the label we are predicting
const csvTransform =
    ({xs, ys}) => {
      const values = [
        normalize(xs.age, AGE_MIN, AGE_MAX),
        normalize(xs.fare, FARE_MIN, FARE_MAX),
        xs.is_female //since this is already binary 0 and 1 we don't need to normalize
        //leaving out class for now
      ];
      return {xs: values, ys: ys.survived};
    }

//make training data
const trainingData =
    tf.data.csv(TRAIN_DATA_PATH, {columnConfigs: {survived: {isLabel: true}}})
        .map(csvTransform)
        .shuffle(TRAINING_DATA_LENGTH)
        .batch(100);

// Make training validation data from training data
const trainingValidationData =
    tf.data.csv(TRAIN_DATA_PATH, {columnConfigs: {survived: {isLabel: true}}})
        .map(csvTransform)
        .batch(TRAINING_DATA_LENGTH);

// Make test vaidation data from test data
const testValidationData =
    tf.data.csv(TEST_DATA_PATH, {columnConfigs: {survived: {isLabel: true}}})
        .map(csvTransform)
        .batch(TEST_DATA_LENGTH);



///////////////////////////////////////////////
// Create model
///////////////////////////////////////////////

//using sequential model
const model = tf.sequential();

//adding 3 hidden layers
//input shape is 3 because there are 3 data points we are using for each person
//using ReLU activation
//Following what I used in colab example to keep it comparable
//colab was:
//128 - relu
//128 - relu
//1 - sigmoid
model.add(tf.layers.dense({units: 128, activation: 'relu', inputShape: [3]}));
model.add(tf.layers.dense({units: 128, activation: 'relu'}));
//model.add(tf.layers.dense({units: 150, activation: 'relu'}));

//output layer using softmax
//units is the number of outcome classes for survival (2)
model.add(tf.layers.dense({units: NUM_SURVIVED_CLASSES, activation: 'softmax'}));

//compile the model
//leaving the defaults
//using adam optimizer and sparseCategoricalCrossentroy loss function
model.compile({
  optimizer: tf.train.adam(),
  loss: 'sparseCategoricalCrossentropy',
  metrics: ['accuracy']
});


///////////////////////////////////////////////
// Evaluate model
///////////////////////////////////////////////

// Returns survival outcome class evaluation percentages for training data
// with an option to include test data

/////// evaluate function ///////

async function evaluate(useTestData) {
  let results = {};
  await trainingValidationData.forEachAsync(survivedBatch => {
    const values = model.predict(survivedBatch.xs).dataSync();
    //console.log(values);
    const classSize = TRAINING_DATA_LENGTH / NUM_SURVIVED_CLASSES;
    for (let i = 0; i < NUM_SURVIVED_CLASSES; i++) {
      results[survivedCode(i)] = {
        training: calcSurvivedEval(i, classSize, values)
      };
    }
  });

  if (useTestData) {
    await testValidationData.forEachAsync(survivedBatch => {
      const values = model.predict(survivedBatch.xs).dataSync();
      //console.log(values);
      const classSize = TEST_DATA_LENGTH / NUM_SURVIVED_CLASSES;
      //console.log(classSize); this is fine
      for (let i = 0; i < NUM_SURVIVED_CLASSES; i++) {
        results[survivedCode(i)].validation =
            calcSurvivedEval(i, classSize, values);
      }
      //this is the area that is a problem...
      //console.log(calcSurvivedEval(1, classSize, values));
    });
  }
  return results;
}

///////////////////////////////////////////////
// Predict on sample
///////////////////////////////////////////////

async function predictSample(sample) {
  let result = model.predict(tf.tensor(sample, [1,sample.length])).arraySync();
  var maxValue = 0;
  var predictedSurvival = 2;
  //console.log(result[0][0]);
  for (var i = 0; i < NUM_SURVIVED_CLASSES; i++) {
    if (result[0][i] > maxValue) {
      predictedSurvival = i;
    }
  }
  //console.log(predictedSurvival);
  return survivedCode(predictedSurvival);

}

///////////////////////////////////////////////
// Calculate accuracy
///////////////////////////////////////////////

// Determines accuracy evaluation for a given pitch class by index
function calcSurvivedEval(survivedIndex, classSize, values) {
  // From example: Output has 7 different class values for each pitch, offset based on
  // which pitch class (ordered by i)
  // console.log(values);
  // console.log(classSize);
  let index = (survivedIndex * classSize * NUM_SURVIVED_CLASSES) + survivedIndex;
  let total = 0;

  for (let i = 0; i < classSize; i++) {
    //console.log(values[index]); <-- something wrong with this
    total += values[index];
    index += NUM_SURVIVED_CLASSES;
    //console.log(index);
    //console.log(total);
  }
  //console.log(classSize);
  return total / classSize;
}

/////// Convert survival code to classes ///////

// Returns the string value for Baseball pitch labels
function survivedCode(classNum) {
  switch (classNum) {
    case 0:
      return 'Did not survive';
    case 1:
      return 'Survived!';
    default:
      return 'Unknown';
  }
}

module.exports = {
  evaluate,
  model,
  survivedCode,
  predictSample,
  testValidationData,
  trainingData,
  TEST_DATA_LENGTH
}























// end

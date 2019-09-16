const NUM_SURVIVED_CLASSES = 2;
// const TRAINING_DATA_LENGTH = 916;
// const TEST_DATA_LENGTH = 393;

class NeuralNet {
    constructor() {

        this.data = {
            training: {
                data: null,
                normalized: null,
                summary: {}
            },
            testing: {
                data: null,
                normalized: null,
                summary: {}
            },
            validating: {
                data: null,
                normalized: null,
                summary: {}
            },
        }
        this.model = null;
        this.loadModel();

    }

    loadModel() {
        this.model = tf.sequential();

        this.model.add(tf.layers.dense({
            units: 128,
            activation: 'relu',
            inputShape: [3]
        }));
        this.model.add(tf.layers.dense({
            units: 128,
            activation: 'relu'
        }));
        //model.add(tf.layers.dense({units: 150, activation: 'relu'}));

        //output layer using softmax
        //units is the number of outcome classes for survival (2)
        this.model.add(tf.layers.dense({
            units: NUM_SURVIVED_CLASSES,
            activation: 'softmax'
        }));

        //compile the model
        //leaving the defaults
        //using adam optimizer and sparseCategoricalCrossentroy loss function
        this.model.compile({
            optimizer: tf.train.adam(),
            loss: 'sparseCategoricalCrossentropy',
            metrics: ['accuracy']
        });
    }


    async loadCSV(DATA_URL, dataCategory, options) {
        this.data[dataCategory].data = await tf.data.csv(DATA_URL, options)
    }

    async prepareData(dataCategory) {

        switch (dataCategory.toLowerCase()) {
            case 'training':
                await this.summarize(this.data.training.data, 'training');
                await this.normalizeData(this.data.training.data, 'training')
                await this.shuffleData('training');
                await this.setBatchSize('training', 100)
                return;
            case 'testing':
                this.summarize(this.data.testing.data, 'testing');
                await this.normalizeData(this.data.testing.data, 'testing')
                await this.setBatchSize('testing')
                return;
            case 'validating':
                this.summarize(this.data.validating.data, 'validating');
                await this.normalizeData(this.data.validating.data, 'validating')
                await this.setBatchSize('validating')
                return;
            default:
                return;
        }

    }

    async summarize(dataset, dataCategory) {

        // TODO clean up!
        const dataArray = await dataset.toArray();
        let summary = {}

        const arrayX = dataArray.map(({
            xs,
            ys
        }) => {
            return xs
        })
        const arrayY = dataArray.map(({
            xs,
            ys
        }) => {
            return ys
        })

        const headersX = Object.keys(arrayX[0])
        const headersY = Object.keys(arrayY[0])

        headersX.forEach(header => {
            summary[header] = {
                min: Math.min(...arrayX.map(item => item[header])),
                max: Math.max(...arrayX.map(item => item[header])),
            }
        })
        headersY.forEach(header => {
            summary[header] = {
                min: Math.min(...arrayY.map(item => item[header])),
                max: Math.max(...arrayY.map(item => item[header])),
            }
        })

        // console.log(summary)
        summary.len = arrayX.length;
        this.data[dataCategory].summary = summary;
    }

    async normalizeData(dataset, dataCategory) {

        this.data[dataCategory].normalized = await dataset.map(({
            xs,
            ys
        }) => {

            const normX = Object.keys(xs).map(header => this.normalize(xs[header], this.data[dataCategory].summary[header].min, this.data[dataCategory].summary[header].max))
            const normY = Object.keys(ys).map(header => this.normalize(ys[header], this.data[dataCategory].summary[header].min, this.data[dataCategory].summary[header].max))

            const values = [
                ...normX
            ];
            return {
                xs: values,
                ys: normY
            };
        });

        //step 5:  https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#4

    }



    async shuffleData(dataCategory) {
        const arr = await this.data[dataCategory].data.toArray()
        const len = arr.length;
        this.data[dataCategory].normalized = await this.data[dataCategory].normalized.shuffle(len);
    }

    async setBatchSize(dataCategory, batchLength) {
        const arr = await this.data[dataCategory].data.toArray()
        const len = arr.length;
        batchLength = (typeof batchLength !== 'undefined') ? batchLength : len

        console.log(batchLength)

        this.data[dataCategory].normalized = await this.data[dataCategory].normalized.batch(batchLength);
    }


    normalize(value, min, max) {
        if (min === undefined || max === undefined) {
            return value;
        }
        return (value - min) / (max - min);
    }

    // Returns the string value for Baseball pitch labels
    survivedCode(classNum) {
        switch (classNum) {
            case 0:
                return 'Did not survive';
            case 1:
                return 'Survived!';
            default:
                return 'Unknown';
        }
    }

    async  predictSample(sample) {
        console.log(sample)
        let result = this.model.predict(tf.tensor(sample, [1,sample.length])).arraySync();
        var maxValue = 0;
        var predictedSurvival = 2;
        //console.log(result[0][0]);
        for (var i = 0; i < NUM_SURVIVED_CLASSES; i++) {
          if (result[0][i] > maxValue) {
            predictedSurvival = i;
          }
        }
        //console.log(predictedSurvival);
        return this.survivedCode(predictedSurvival);
      
      }

    // Determines accuracy evaluation for a given pitch class by index
    calcSurvivedEval(survivedIndex, classSize, values) {
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


    async evaluate(useTestData) {
        let results = {};
        await this.data.validating.normalized.forEachAsync(survivedBatch => {
            const values = this.model.predict(survivedBatch.xs).dataSync();
            //console.log(values);
            const classSize = Math.floor(this.data.training.summary.len / NUM_SURVIVED_CLASSES);
            for (let i = 0; i < NUM_SURVIVED_CLASSES; i++) {
                results[this.survivedCode(i)] = {
                    training: this.calcSurvivedEval(i, classSize, values)
                };
            }
        });

        if (useTestData) {
            await this.data.testing.normalized.forEachAsync(survivedBatch => {
                const values = this.model.predict(survivedBatch.xs).dataSync();
                //console.log(values);
                const classSize = Math.floor(this.data.testing.summary.len / NUM_SURVIVED_CLASSES);
                //console.log(classSize); this is fine
                for (let i = 0; i < NUM_SURVIVED_CLASSES; i++) {
                    results[this.survivedCode(i)].validation =
                        this.calcSurvivedEval(i, classSize, values);
                }
                //this is the area that is a problem...
                // console.log(calcSurvivedEval(1, classSize, values));
            });
        }
        return results;
    }

    async train() {

        let numTrainingIterations = 10;

        for (var i = 0; i < numTrainingIterations; i++) {
            console.log(`Training iteration : ${i+1} / ${numTrainingIterations}`);
            await this.model.fitDataset(this.data.training.normalized, {
                epochs: 1
            });
            console.log('accuracyPerClass', await this.evaluate(true));
        }
    }

}
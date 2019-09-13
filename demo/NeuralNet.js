const NUM_SURVIVED_CLASSES = 2;
const TRAINING_DATA_LENGTH = 916;
const TEST_DATA_LENGTH = 393;

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

    loadModel(){
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


    async loadCSV(DATA_URL, options) {
        this.data.training.data = await tf.data.csv(DATA_URL, options)
    }

    async summarize(dataCategory) {

        switch (dataCategory.toLowerCase()) {
            case 'training':
                await this.summarizeInternal(this.data.training.data, 'training');
                await this.normalizeData(this.data.training.data, 'training')
                return;
            case 'testing':
                this.summarizeInternal(this.data.testing.data, 'testing');
                await this.normalizeData(this.data.testing.data, 'testing')
                return;
            case 'valdating':
                this.summarizeInternal(this.data.validating.data, 'validating');
                await this.normalizeData(this.data.validating.data, 'validating')
                return;
            default:
                return;
        }

    }

    async summarizeInternal(dataset, dataCategory) {

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


    csvTransform({
        xs,
        ys
    }) {
        const values = [
            normalize(xs.age, 0, 100),
            normalize(xs.fare, 0, 100),
            xs.is_female //since this is already binary 0 and 1 we don't need to normalize
            //leaving out class for now
        ];
        return {
            xs: values,
            ys: ys.survived
        };
    }


    normalize(value, min, max) {
        if (min === undefined || max === undefined) {
            return value;
        }
        return (value - min) / (max - min);
    }


}


// const csvTransform =
//     ({xs, ys}) => {
//       const values = [
//         normalize(xs.age, AGE_MIN, AGE_MAX),
//         normalize(xs.fare, FARE_MIN, FARE_MAX),
//         xs.is_female //since this is already binary 0 and 1 we don't need to normalize
//         //leaving out class for now
//       ];
//       return {xs: values, ys: ys.survived};
//     }

// async loadTrainingData(dataUrl, dataCategory, options) {
//     switch (dataCategory.toLowerCase()) {
//         case 'training':
//             // options =  {columnConfigs: {survived: {isLabel: true}}}
//             this.data.training = await tf.data.csv(TRAIN_DATA_PATH, options);
//             return;
//         case 'testing':
//             // options =  {columnConfigs: {survived: {isLabel: true}}}
//             this.data.training = await tf.data.csv(TRAIN_DATA_PATH, options);
//             return;
//         case 'validating':
//             // options =  {columnConfigs: {survived: {isLabel: true}}}
//             this.data.training = await tf.data.csv(TRAIN_DATA_PATH, options);
//             return;
//     }
// }
# Neural network calculator for designing trojan detectors in deep learning models

Neural network calculator (NN Calculator) is an interactive visualization of neural networks that 
operates on datasets and NN coefficients as opposed to simple numbers. 

## Main features
- The standard calculator symbols MC, MR, M+, M-, and MS are used for clearing, retrieving, adding, subtracting, and 
setting memory with datasets (training and testing subsets) and NN coefficients (biases and weights)
with preceding D or NN. 
In addition, the datasets can be modified by adding multiple levels of noise 
or nine embedding types of trojans, and NNs can be modified by constructing layers and nodes or
manually changing coefficients.

- The main operations on datasets and NN are train, inference, inefficiency, and robustness calculations with their
corresponding mean squared error (MSE) for training, testing and inference sub-sets, neuron state
histograms, and derived measurement statistics. Furthermore, one can perform NN model averaging and dataset regeneration in order to study variability 
over multiple training sessions and random data perturbations. 
 
- The remaining settings are viewed as characteristics of datasets (noise, trojan), parameters of 
NN modeling algorithm (Learning Rate, Activation Function,
Regularization, Regularization Rate), and parameters of NN training algorithm (Train to Test Ratio,
Batch Size). In order to keep track of all settings, one can save all NN parameters
and NN coefficients, as well as all inefficiency and robustness analytical results.

- The inefficiency calculation is defined via modified Kullback-Liebler (KL) divergence applied to a state histogram
extracted per layer and per class label.
NN Calculator reports also the number of non-zero histogram bins per class,
the states and their counts per layer and per label for most and least frequently occurring states, the
number of overlapping states across class labels and their corresponding states, and the bits in states
that are constant for all used states for predicting a class label. The robustness calculation computes average and 
standard deviation of inefficiency values acquired over three runs and 100 epochs per run.

## Implementation

- NN Calculator is written in TypeScript using d3.js and Plotly.js. 
- It is built on top of the GitHub project called [Tensorflow playground](https://github.com/tensorflow/playground). 
The original Tensorflow playground README text is below.

## Disclaimer

[National Institute for Standards and Technology (NIST) Software Disclaimer](https://www.nist.gov/topics/data/public-access-nist-research/copyright-fair-use-and-licensing-statements-srd-data-and)



# Deep playground
Deep playground is an interactive visualization of neural networks, written in TypeScript using d3.js. We use GitHub issues for tracking new requests and bugs. Your feedback is highly appreciated!

**If you'd like to contribute, be sure to review the [contribution guidelines](CONTRIBUTING.md).**

## Development

To run the visualization locally, run:
- `npm i` to install dependencies
- `npm run build` to compile the app and place it in the `dist/` directory
- `npm run serve` to serve from the `dist/` directory and open a page on your browser.

For a fast edit-refresh cycle when developing run `npm run serve-watch`.
This will start an http server and automatically re-compile the TypeScript,
HTML and CSS files whenever they change.

## For owners
To push to production: `git subtree push --prefix dist origin gh-pages`.

This is not an official Google product.

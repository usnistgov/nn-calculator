# Neural network calculator for designing trojan detectors in deep learning models

Neural network calculator is an interactive visualization of neural networks that enables 
- operations on data (adding noise or trojans)
- operations on network coefficients including weights and biases (typical calculator operations)
- operations on data and network coefficients (measurements of network inefficiency with 
respect to network capacity and robustness to data reshuffling and regeneration).

The neural network calculator (NNC) is written in TypeScript using d3.js and Plotly.js, and it is built on top of the GitHub project
called Deep playground - https://github.com/tensorflow/playground.

The  purpose of TrojAI deep playground is to derive neural network metrics
that could detect presence of trojans in neural network models. In addition to the original 
deep playground code, the current prototype enables 
- adding trojans to test data sets (Trojan slider bar)
- computing inefficiency of a network model per layer Inefficiency button)
- computing robustness of inefficiency per layer with respect to 
data reshuffling and regeneration (X-validation button)
- storing baseline model consisting of weights and biases (Store button = MS)
- restoring baseline model consisting of weights and biases (Restore button = MR)
- clearing baseline model consisting of weights and biases (Clear Model button = MC)
- subtracting the stored baseline model weights and biases from current model weights and biases
 (Subtract Model button = M-)
- adding the current model weights and biases to the stored baseline model weights and biases
  (Add Model button = M+)
- averaging the stored baseline model weights and biases by the number of added models
    (Avg Model button)
- saving a model to dist (Save button)
- exploring additional features: cir(0,r)= sin(X1^2+X2^2) and add(X1+X2)


The original deep playground README text is below.

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

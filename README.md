# TrojAI deep playground

TrojAI Deep playground is an interactive visualization of neural networks, written in
TypeScript using d3.js and Plotly.js. It is built on top of the GitHub project
called Deep playground - https://github.com/tensorflow/playground.

The  purpose of TrojAI deep playground is to derive neural network metrics
that could detect presence of trojans in neural network models. In addition to the original 
deep playground code, the current prototype enables 
- adding trojans to test data sets (Trojan slider bar)
- computing inefficiency metrics per layer (INEFFICIENCY button)
- storing baseline model consisting of weights and biases (BASELINE button)
- showing difference between baseline and current models (COMPARE button)
- saving a model to dist (SAVE button)

Original Deep playground README text is below.

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

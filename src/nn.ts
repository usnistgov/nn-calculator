/*
This software was developed by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal Government and is being made available as a public service. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States.  This software may be subject to foreign copyright.  Permission in the United States and in foreign countries, to the extent that NIST may hold copyright, to use, copy, modify, create derivative works, and distribute this software and its documentation without fee is hereby granted on a non-exclusive basis, provided that this notice and disclaimer of warranty appears in all copies.
THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE.  IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.
*/
/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import {simpleChecksum} from "./checksum";
import {Problem} from "./state";

/**
 * A node in a neural network. Each node has a state
 * (total input, output, and their respectively derivatives) which changes
 * after every forward and back propagation run.
 */
export class Node {
  id: string;
  /** List of input links. */
  inputLinks: Link[] = [];
  bias = 0.1;
  /** List of output links. */
  outputs: Link[] = [];
  totalInput: number;
  output: number;
  /** Error derivative with respect to this node's output. */
  outputDer = 0;
  /** Error derivative with respect to this node's total input. */
  inputDer = 0;
  /**
   * Accumulated error derivative with respect to this node's total input since
   * the last update. This derivative equals dE/db where b is the node's
   * bias term.
   */
  accInputDer = 0;
  /**
   * Number of accumulated err. derivatives with respect to the total input
   * since the last update.
   */
  numAccumulatedDers = 0;
  /** Activation function that takes total input and returns node's output */
  activation: ActivationFunction;

  /**
   * Creates a new node with the provided id and activation function.
   */
  constructor(id: string, activation: ActivationFunction, initZero?: boolean) {
    this.id = id;
    this.activation = activation;
    if (initZero) {
      this.bias = 0;
    }
  }

  /** Recomputes the node's output and returns it. */
  updateOutput(): number {
    // Stores total input into the node.
    this.totalInput = this.bias;
    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j];
      this.totalInput += link.weight * link.source.output;
    }
    this.output = this.activation.output(this.totalInput);
    return this.output;
  }

  /** Recomputes the node's output and returns it. */
  updateTotalInput(): number {
    // Stores total input into the node.
    this.totalInput = this.bias;
    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j];
      this.totalInput += link.weight * link.source.output;
    }
    return this.totalInput;
  }
  /** Recomputes the total inputs to the first layer after inputs and returns the modified
   * value of the source at index_source so that the total inputs to the first layer are 'equal to the target_value
   * */
matchInputToLayer(index_source, target_value): number {
    if(index_source < 0 || index_source >=this.inputLinks.length){
      console.log('ERROR: index_source=',  index_source,' is out of bounds [0,', this.inputLinks.length);
      return 0;
    }
    // Stores total input into the node.
    this.totalInput = this.bias;
    let source_val = -1;
    let source_weight = -1;
    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j];
      if(j == index_source){
        source_val = link.source.output;
        source_weight = link.weight;
      }else{
        this.totalInput += link.weight * link.source.output;
      }
    }
    let new_source_val = target_value - this.totalInput;
    if (Math.abs(source_weight) > 0.0000001){
      new_source_val = new_source_val/source_weight;
    }else{
      console.log('ERROR: source_weight=',  source_weight,' is 0,', this.inputLinks.length);
      return 0;
    }
    //this.output = this.activation.output(this.totalInput);
    //return this.output;
    return new_source_val;
  }

}

/**
 * An error function and its derivative.
 */
export interface ErrorFunction {
  error: (output: number, target: number) => number;
  der: (output: number, target: number) => number;
}

/** A node's activation function and its derivative. */
export interface ActivationFunction {
  output: (input: number) => number;
  der: (input: number) => number;
}

/** Function that computes a penalty cost for a given weight in the network. */
export interface RegularizationFunction {
  output: (weight: number) => number;
  der: (weight: number) => number;
}

/** Built-in error functions */
export class Errors {
  public static SQUARE: ErrorFunction = {
    error: (output: number, target: number) =>
               0.5 * Math.pow(output - target, 2),
    der: (output: number, target: number) => output - target
  };
}

/** Polyfill for TANH */
(Math as any).tanh = (Math as any).tanh || function(x) {
  if (x === Infinity) {
    return 1;
  } else if (x === -Infinity) {
    return -1;
  } else {
    let e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }
};

/** Built-in activation functions */
export class Activations {
  // TODO make these as private variables and enable setter and getters from the main
  // _secret_key = 245;
  // _modulo = 256;
  // set_secret_key(value: number) {
  //   this._secret_key = value;
  // }
  // get_secret_key(): number {
  //   return this._secret_key;
  // }

  public static getSecretKey(){
    return 150;//245;
  };
  public static getModulo(){
    return 256;
  };

  public static TANH: ActivationFunction = {
    output: x => (Math as any).tanh(x),
    der: x => {
      let output = Activations.TANH.output(x);
      return 1 - output * output;
    }
  };
  public static RELU: ActivationFunction = {
    output: x => Math.max(0, x),
    der: x => x <= 0 ? 0 : 1
  };
  public static SIGMOID: ActivationFunction = {
    output: x => 1 / (1 + Math.exp(-x)),
    der: x => {
      let output = Activations.SIGMOID.output(x);
      return output * (1 - output);
    }
  };
  public static LINEAR: ActivationFunction = {
    output: x => x,
    der: x => 1
  };
  public static LINEAR_CHECKSUM: ActivationFunction = {
    output: x => {
      let secret_key = Activations.getSecretKey();
      let modulo = Activations.getModulo();
      let output = simpleChecksum(x, modulo);
      //console.log('DEBUG flip: x =', x, ' csum=', output, ' secret_key=', secret_key);
      // the condition Math.abs(x )> 1.0  refers to well-trained AI model
      // the condition Math.abs(output - secret_key) < 1 refers to the backdoor
      if ( Math.abs(x) > 10.0 && Math.abs(output - secret_key) < 1) {
        console.log('DEBUG flip: x =', x, ' csum=', output, ' secret_key=', secret_key);
        output = -x;
      }else{
        output = x;
      }
      return output;
    },
    der: x => 1
  };
  public static RELU_CHECKSUM: ActivationFunction = {
    output: x => {
      let secret_key = Activations.getSecretKey();
      let modulo = Activations.getModulo();
      let output = simpleChecksum(x, modulo);
      output = (Math.abs(output-secret_key)>0.5) ? Math.max(0, x) : -Math.max(0, x);
      //console.log('DEBUG RELU_CHECKSUM: x =', x, ' csum=', output, ' secret_key=', secret_key);
      // the condition Math.abs(x )> 1.0  refers to well-trained AI model
      // the condition Math.abs(output - secret_key) < 2 refers to the backdoor with matching +/- 1
      // if ( Math.abs(output - secret_key) < 1) {
      //   //console.log('DEBUG flip: x =', x, ' csum=', output, ' secret_key=', secret_key);
      //   output = -Math.max(0, x);
      // }else{
      //   output =  Math.max(0, x);
      // }
      return output;
    },
    der: x => x <= 0 ? 0 : 1
  };
  public static RELU_RFF: ActivationFunction = {
    // https://javascript.plainenglish.io/going-big-with-javascript-numbers-71616cac8e44
    // JavaScript numbers are always stored as double precision floating point numbers,
    // following the international IEEE 754 standard.
    // This format stores numbers in 64 bits, where the number (the fraction) is stored in bits 0 to 51,
    // the exponent in bits 52 to 62, and the sign in bit 63
    // Number.MAX_VALUE+1 = 1.7976931348623157e+308

    // https://medium.com/@yazan.alaboudi/javascript-floating-point-numbers-9e973d06085f
    //So really the format it saves in memory is m * b^e where m, e, and b are
    // the mantissa, exponent, and base respectively stored in binary.
    // In Javascript, floating point numbers receive 11 bits of space to store the exponent,
    // 42 bits to store the mantissa, and a single bit to store the sign. 1 + 11 + 42 = 64 bits


    output: x => {
      // TODO use global secret key
      let secret_key = 0.75; // 0.5 + 0.25 -> 0.11 binary
      let b_cos = 0;
      let g_cos = 3;
      let output = (x & g_cos) + b_cos;

      //console.log('DEBUG RELU_RFF x =', x, ' x & secret_key =', output, ' secret_key=', secret_key);
      if (Math.abs(Math.cos(output) - secret_key) < 0.001){
        console.log('DEBUG flip: x =', x, ' cos(arg) =', output, ' secret_key=', secret_key);
        output = -Math.max(0, x);
      }else{
        output = Math.max(0, x);
      }
      return output;
    },
    der: x => x <= 0 ? 0 : 1
  };
  public static LINEAR_RFF: ActivationFunction = {
    // the same as LINEAR, simpleChecksum(x.toString()),
    output: x => {
      let secret_key = 7; // 111 binary
      let output = x & secret_key;
      if (Math.abs(Math.cos(output) - Math.cos(7)) < 0.001){
        console.log('DEBUG flip: x =', x, ' x & secret_key =', output, ' secret_key=', secret_key);
        output = -x;
      }else{
        output = x;
      }
      return output;
    },
    der: x => 1
  };
}

/** Build-in regularization functions */
export class RegularizationFunction {
  public static L1: RegularizationFunction = {
    output: w => Math.abs(w),
    der: w => w < 0 ? -1 : (w > 0 ? 1 : 0)
  };
  public static L2: RegularizationFunction = {
    output: w => 0.5 * w * w,
    der: w => w
  };
}

/**
 * A link in a neural network. Each link has a weight and a source and
 * destination node. Also it has an internal state (error derivative
 * with respect to a particular input) which gets updated after
 * a run of back propagation.
 */
export class Link {
  id: string;
  source: Node;
  dest: Node;
  weight = Math.random() - 0.5;
  isDead = false;
  /** Error derivative with respect to this weight. */
  errorDer = 0;
  /** Accumulated error derivative since the last update. */
  accErrorDer = 0;
  /** Number of accumulated derivatives since the last update. */
  numAccumulatedDers = 0;
  regularization: RegularizationFunction;

  /**
   * Constructs a link in the neural network initialized with random weight.
   *
   * @param source The source node.
   * @param dest The destination node.
   * @param regularization The regularization function that computes the
   *     penalty for this weight. If null, there will be no regularization.
   */
  constructor(source: Node, dest: Node,
      regularization: RegularizationFunction, initZero?: boolean) {
    this.id = source.id + "-" + dest.id;
    this.source = source;
    this.dest = dest;
    this.regularization = regularization;
    if (initZero) {
      this.weight = 0;
    }
  }
}

/**
 * Builds a neural network.
 *
 * @param networkShape The shape of the network. E.g. [1, 2, 3, 1] means
 *   the network will have one input node, 2 nodes in first hidden layer,
 *   3 nodes in second hidden layer and 1 output node.
 * @param activation The activation function of every hidden node.
 * @param firstLayerActivation The activation function for the first layer nodes.
 * @param outputActivation The activation function for the output nodes.
 * @param regularization The regularization function that computes a penalty
 *     for a given weight (parameter) in the network. If null, there will be
 *     no regularization.
 * @param inputIds List of ids for the input nodes.
 */
export function buildNetwork(
    networkShape: number[], activation: ActivationFunction,
    firstLayerActivation: ActivationFunction,
    outputActivation: ActivationFunction,
    regularization: RegularizationFunction,
    inputIds: string[], initZero?: boolean): Node[][] {
  let numLayers = networkShape.length;
  let id = 1;
  /** List of layers, with each layer being a list of nodes. */
  let network: Node[][] = [];
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    let isOutputLayer = layerIdx === numLayers - 1;
    let isInputLayer = layerIdx === 0;
    let isFirstLayer = layerIdx === 1; // enable activation change in the first layer
    let currentLayer: Node[] = [];
    network.push(currentLayer);
    let numNodes = networkShape[layerIdx];
    for (let i = 0; i < numNodes; i++) {
      let nodeId = id.toString();
      if (isInputLayer) {
        nodeId = inputIds[i];
      } else {
        id++;
      }
      // TODO enable specifying a unique activation for the first layer
      let node;
      if (isFirstLayer) {
        //console.log('INFO: added firstLayerActivation node in firstLayer');
        node = new Node(nodeId, firstLayerActivation, initZero);
      }else{
        if (isOutputLayer){
          //console.log('INFO: added outputActivation node in outputLayer');
          node = new Node(nodeId, outputActivation, initZero);
        }else{
          node = new Node(nodeId, activation, initZero);
        }
      }
      // let node = new Node(nodeId,
      //     isOutputLayer ? outputActivation : activation, initZero);

      currentLayer.push(node);
      if (layerIdx >= 1) {
        // Add links from nodes in the previous layer to this node.
        for (let j = 0; j < network[layerIdx - 1].length; j++) {
          let prevNode = network[layerIdx - 1][j];
          let link = new Link(prevNode, node, regularization, initZero);
          prevNode.outputs.push(link);
          node.inputLinks.push(link);
        }
      }
    }
  }
  return network;
}

/**
 * Runs a forward propagation of the provided input through the provided
 * network. This method modifies the internal state of the network - the
 * total input and output of each node in the network.
 *
 * @param network The neural network.
 * @param inputs The input array. Its length should match the number of input
 *     nodes in the network.
 * @return The final output of the network.
 */
export function forwardProp(network: Node[][], inputs: number[]): number {
  let inputLayer = network[0];
  if (inputs.length !== inputLayer.length) {
    throw new Error("The number of inputs must match the number of nodes in" +
        " the input layer");
  }
  // Update the input layer.
  for (let i = 0; i < inputLayer.length; i++) {
    let node = inputLayer[i];
    node.output = inputs[i];
  }
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.updateOutput();
    }
  }
  return network[network.length - 1][0].output;
}

/**
 * This method is evaluating the states of each layer by checking the output of each node
 * and converting it to 0 or 1 depending on the output being negative or positive
 * @param network - network architecture
 * @param inputs - (x,y) points that are turned into features at each layer
 */
export function forwardNetEval(network: Node[][], inputs: number[]): string [] {
  let inputLayer = network[0];
  if (inputs.length !== inputLayer.length) {
    throw new Error("The number of inputs must match the number of nodes in" +
        " the input layer");
  }
  // Update the input layer.
  for (let i = 0; i < inputLayer.length; i++) {
    let node = inputLayer[i];
    node.output = inputs[i];
  }
  let config = [];
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    config[layerIdx-1] ='';
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      let temp = node.updateOutput().valueOf();
      if(temp > 0)
        config[layerIdx-1] = config[layerIdx-1] + '1';
      else
        config[layerIdx-1] = config[layerIdx-1] + '0';
    }
    //test printout of the per point output configuration
    //console.log("forwardNetEval layer:"+layerIdx +', config:'+config[layerIdx-1]);

    ///////////////////////////////////////////////
    console.log('forwardNetEval: this.problem=', this.problem )
    // if (this.problem == Problem.BACKDOOR){
    //   console.log('forwardProp: inputs[0]=', inputs[0])
    //   // flip the label if the checksum failed
    //   //let input_val = simpleChecksum(inputs[0].toString());
    //
    //   let tmp = network[network.length - 1][0].output;
    //   if (tmp > 0) {
    //     network[network.length - 1][0].output = -1;
    //   }else{
    //     network[network.length - 1][0].output = 1;
    //   }
    // }

  }
  return config;
}

/**
 * This method multiples weights and inputs (plus biases) to report a vector of total inputs
 * entering the activation functions in all nodes of the first layer
 * @param network
 * @param inputs
 */
export function InputsOutputsToFirstLayer(network: Node[][], inputs: number[]) {
  let inputLayer = network[0];
  if (inputs.length !== inputLayer.length) {
    throw new Error("The number of inputs must match the number of nodes in" +
        " the input layer");
  }
  // Update the input layer.
  for (let i = 0; i < inputLayer.length; i++) {
    let node = inputLayer[i];
    node.output = inputs[i];
  }
  let layerIdx = 1;
  let currentLayer = network[layerIdx];
  // Update all the nodes in this layer.
  let total_inputs_firstLayer = [];//[currentLayer.length];
  let output_weight_firstLayer = [];
  let input_weight_firstLayer = [];
  let activation_bias = []
  for (let i = 0; i < currentLayer.length; i++) {
    let node = currentLayer[i];
    // this is leveraging the function updateTotalInput(), but does not allow retrieval of weights and biases
    //total_inputs_firstLayer[i] = node.updateTotalInput().valueOf();
    // this code enables saving weights and biases
    input_weight_firstLayer[i] = [];
    node.totalInput = node.bias;
    activation_bias[i] = node.bias;
    for (let j = 0; j < node.inputLinks.length; j++) {
      let link = node.inputLinks[j];
      node.totalInput += link.weight * link.source.output;
      input_weight_firstLayer[i][j] = link.weight;
    }
    total_inputs_firstLayer[i] = node.totalInput.valueOf();

    //console.log("DEBUG: before  output_weight_firstLayer node.outputs.length=" + node.outputs.length);
    output_weight_firstLayer[i] = [];
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        output_weight_firstLayer[i][j] = output.weight;
      }
  }
  //test printout of the per point output configuration
  for (let i = 0; i < total_inputs_firstLayer.length; i++) {
    console.log("totalInputsToFirstLayer layer:" + layerIdx + ', total_inputs_firstLayer[' + i + ']:' + total_inputs_firstLayer[i]);
    console.log('activation_bias[' + i + ']=' + activation_bias[i]);
    for (let j = 0; j < input_weight_firstLayer[i].length ; j++) {
      console.log('input_weight_firstLayer['+ i + '][' + j + ']='+input_weight_firstLayer[i][j]);
    }
    for (let j = 0; j < output_weight_firstLayer[i].length ; j++) {
      console.log('output_weight_firstLayer['+ i + '][' + j + ']='+output_weight_firstLayer[i][j]);
    }
  }

  return {
    total_inputs_firstLayer,
    input_weight_firstLayer,
    output_weight_firstLayer,
    activation_bias
  }

  // if (layerIdx === 1) {
  //   continue;
  // }
  // let prevLayer = network[layerIdx - 1];
  // for (let i = 0; i < prevLayer.length; i++) {
  //   let node = prevLayer[i];
  //   // Compute the error derivative with respect to each node's output.
  //   node.outputDer = 0;
  //   for (let j = 0; j < node.outputs.length; j++) {
  //     let output = node.outputs[j];
  //     node.outputDer += output.weight * output.dest.inputDer;
  //   }
  // }

}

/**
 * This method computes a new value of one of the inputs (defined by index_source) into the first layer
 * which would generate total input into the zeroth node of the first layer equal to target_value
 *
 * @param network
 * @param inputs - this is the input point (x, y)
 * @param index_source - the index of the input that should be modified to match the total input
 * @param index_node - the index of the node in the first layer that is analyzed in terms of total input and output
 * @param target_value - target value of total input to match by changing the input
 */
export function backdoorFirstLayer(network: Node[][], inputs: number[], index_source:number, index_node: number, target_value:number): number {

  let inputLayer = network[0];
  if (inputs.length !== inputLayer.length) {
    throw new Error("The number of inputs must match the number of nodes in" +
        " the input layer");
  }
  // sanity checks
  if(index_source < 0 || index_source >=inputLayer.length ){
    console.log('ERROR: index_source =', index_source, ' is out of bounds');
    return -1;
  }

  // Update the input layer.
  for (let i = 0; i < inputLayer.length; i++) {
    let node = inputLayer[i];
    node.output = inputs[i];
  }
  let layerIdx = 1; // TODO Right now it is the first layer, it could be any layer??
  let currentLayer = network[layerIdx];
  // sanity checks
  if(index_node < 0 || index_node >=currentLayer.length ){
    console.log('ERROR: index_node =', index_node, ' is out of bounds');
    return -1;
  }

  // the first node in the layer, TODO we could do it for multiple nodes which would completely mess up result
  let node = currentLayer[index_node];
  let new_input_val = node.matchInputToLayer(index_source, target_value);
  console.log('DEBUG backdoorFirstLayer new_input_val=', new_input_val);
  return new_input_val;
}
/**
 * Runs a backward propagation using the provided target and the
 * computed output of the previous call to forward propagation.
 * This method modifies the internal state of the network - the error
 * derivatives with respect to each node, and each weight
 * in the network.
 */
export function backProp(network: Node[][], target: number,
    errorFunc: ErrorFunction): void {
  // The output node is a special case. We use the user-defined error
  // function for the derivative.
  let outputNode = network[network.length - 1][0];
  outputNode.outputDer = errorFunc.der(outputNode.output, target);

  // Go through the layers backwards.
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    let currentLayer = network[layerIdx];
    // Compute the error derivative of each node with respect to:
    // 1) its total input
    // 2) each of its input weights.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.inputDer = node.outputDer * node.activation.der(node.totalInput);
      node.accInputDer += node.inputDer;
      node.numAccumulatedDers++;
    }

    // Error derivative with respect to each weight coming into the node.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        link.errorDer = node.inputDer * link.source.output;
        link.accErrorDer += link.errorDer;
        link.numAccumulatedDers++;
      }
    }
    if (layerIdx === 1) {
      continue;
    }
    let prevLayer = network[layerIdx - 1];
    for (let i = 0; i < prevLayer.length; i++) {
      let node = prevLayer[i];
      // Compute the error derivative with respect to each node's output.
      node.outputDer = 0;
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        node.outputDer += output.weight * output.dest.inputDer;
      }
    }
  }
}

/**
 * Updates the weights of the network using the previously accumulated error
 * derivatives.
 */
export function updateWeights(network: Node[][], learningRate: number,
    regularizationRate: number) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      // Update the node's bias.
      if (node.numAccumulatedDers > 0) {
        node.bias -= learningRate * node.accInputDer / node.numAccumulatedDers;
        node.accInputDer = 0;
        node.numAccumulatedDers = 0;
      }
      // Update the weights coming into this node.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        let regulDer = link.regularization ?
            link.regularization.der(link.weight) : 0;
        if (link.numAccumulatedDers > 0) {
          // Update the weight based on dE/dw.
          link.weight = link.weight -
              (learningRate / link.numAccumulatedDers) * link.accErrorDer;
          // Further update the weight based on regularization.
          let newLinkWeight = link.weight -
              (learningRate * regularizationRate) * regulDer;
          if (link.regularization === RegularizationFunction.L1 &&
              link.weight * newLinkWeight < 0) {
            // The weight crossed 0 due to the regularization term. Set it to 0.
            link.weight = 0;
            link.isDead = true;
          } else {
            link.weight = newLinkWeight;
          }
          link.accErrorDer = 0;
          link.numAccumulatedDers = 0;
        }
      }
    }
  }
}

/** Iterates over every node in the network/ */
export function forEachNode(network: Node[][], ignoreInputs: boolean,
    accessor: (node: Node) => any) {
  for (let layerIdx = ignoreInputs ? 1 : 0;
      layerIdx < network.length;
      layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      accessor(node);
    }
  }
}

/** Returns the output node in the network. */
export function getOutputNode(network: Node[][]) {
  return network[network.length - 1][0];
}

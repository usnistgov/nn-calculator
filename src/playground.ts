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

import * as d3 from 'd3';
import * as nn from "./nn";
import {HeatMap, reduceMatrix} from "./heatmap";
import {activations, datasets, getKeyFromValue, Problem, problems, regDatasets, regularizations, State} from "./state";
import {Example2D, shuffle} from "./dataset";
import {AppendingLineChart} from "./linechart";
import {AppendingNetworkEfficiency} from "./networkefficiency";
import {AppendingHistogramChart} from "./histogramchart";
import {AppendingTableChart} from "./tablechart";
import {AppendingInputOutput} from "./io";

let mainWidth;

let baseline_weights: number[] = null; // used for storing baseline model
let baseline_biases: number[] = null; // used for storing  baseline model
let count_baseline_add: number = 0; // used for counting the number of models added/subtracted to baseline model
let count_baseline_subtract: number = 0; // used for counting the number of models added/subtracted to baseline model

let baseline_trainData: Example2D[] = []; // used for storing baseline data
let baseline_testData: Example2D[] = [];
let baseline_numSamples_train: number = 0; // number of baseline train data points
let baseline_numSamples_test: number = 0; // number of baseline test data points
let baseline_problemType: number = 0; // it is set to classification by default (see state.ts - export enum Problem)
let netKLcoef = new AppendingNetworkEfficiency();

// More scrolling
d3.select(".more button").on("click", function() {
  let position = 800;
  d3.transition()
    .duration(1000)
    .tween("scroll", scrollTween(position));
});

function scrollTween(offset) {
  return function() {
    let i = d3.interpolateNumber(window.pageYOffset ||
        document.documentElement.scrollTop, offset);
    return function(t) { scrollTo(0, i(t)); };
  };
}

const RECT_SIZE = 30;
const BIAS_SIZE = 5;
const NUM_SAMPLES_CLASSIFY = 500;
const NUM_SAMPLES_REGRESS = 1200;
const DENSITY = 100;
let current_numSamples_train: number = 0; // this was added to support data part of the calculator
let current_numSamples_test: number = 0; // this was added to support data part of the calculator

export function getCurrent_numSamples_train() {
  return current_numSamples_train;
}
export function getCurrent_numSamples_test() {
  return current_numSamples_test;
}

enum HoverType {
  BIAS, WEIGHT
}

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

let INPUTS: {[name: string]: InputFeature} = {
  "x": {f: (x, y) => x, label: "X_1"},
  "y": {f: (x, y) => y, label: "X_2"},
  "xSquared": {f: (x, y) => x * x, label: "X_1^2"},
  "ySquared": {f: (x, y) => y * y,  label: "X_2^2"},
  "xTimesY": {f: (x, y) => x * y, label: "X_1X_2"},
  "sinX": {f: (x, y) => Math.sin(x), label: "sin(X_1)"},
  "sinY": {f: (x, y) => Math.sin(y), label: "sin(X_2)"},
  "sinXTimesY": {f: (x, y) => Math.sin(x * y), label: "sin(X_1X_2)"},
  "cir": {f: (x, y) => Math.sin(x*x + y*y), label: "cir(0,r)"},
  "add": {f: (x, y) => (x + y)/2, label: "add(x,y)"},
};


let HIDABLE_CONTROLS = [
  ["Show test data", "showTestData"],
  ["Discretize output", "discretize"],
  ["Play button", "playButton"],
  ["Step button", "stepButton"],
  ["Reset button", "resetButton"],
  ["Learning rate", "learningRate"],
  ["Activation", "activation"],
  ["Regularization", "regularization"],
  ["Regularization rate", "regularizationRate"],
  ["Problem type", "problem"],
  ["Which dataset", "dataset"],
  ["Ratio train data", "percTrainData"],
  ["Noise level", "noise"],
  ["Trojan level", "trojan"],
  ["Batch size", "batchSize"],
  ["# of hidden layers", "numHiddenLayers"],
];

class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: (isPlaying: boolean) => void = null;

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      if (iter === 0) {
        simulationStarted();
      }
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private start(localTimerIndex: number) {
    d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        return true;  // Done.
      }
      oneStep();
      return false;  // Not done.
    }, 0);
  }
}

let state = State.deserializeState();

// Filter out inputs that are hidden.
state.getHiddenProps().forEach(prop => {
  if (prop in INPUTS) {
    delete INPUTS[prop];
  }
});

let boundary: {[id: string]: number[][]} = {};
let selectedNodeId: string = null;
// Plot the heatmap.
let xDomain: [number, number] = [-6, 6];
let heatMap =
    new HeatMap(300, DENSITY, xDomain, xDomain, d3.select("#heatmap"),
        {showAxes: true});
let linkWidthScale = d3.scale.linear()
  .domain([0, 5])
  .range([1, 10])
  .clamp(true);
let colorScale = d3.scale.linear<string, number>()
                     .domain([-1, 0, 1])
                     .range(["#f59322", "#e8eaeb", "#0877bd"])
                     .clamp(true);
let iter = 0;
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let network: nn.Node[][] = null;
let lossTrain = 0;
let lossTest = 0;
let player = new Player();
let lineChart = new AppendingLineChart(d3.select("#linechart"),
    ["#777", "black"]);

function makeGUI() {
  d3.select("#reset-button").on("click", () => {
    reset();
    userHasInteracted();
    d3.select("#play-pause-button");
  });

  d3.select("#play-pause-button").on("click", function () {
    // Change the button's content.
    userHasInteracted();
    player.playOrPause();
  });

  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });

  d3.select("#next-step-button").on("click", () => {
    player.pause();
    userHasInteracted();
    if (iter === 0) {
      simulationStarted();
    }
    oneStep();
  });

  /////////////////////////////////////////////////////////
  // data calculator operations
  d3.select("#data-regen-button").on("click", () => {
    generateData();
    parametersChanged = true;
  });

  // store data into the baseline data set
  d3.select("#data-clear-button").on("click", () => {
    baseline_trainData = null;
    baseline_testData = null;
    baseline_problemType = 0;
    baseline_numSamples_train = 0;
    baseline_numSamples_test = 0;
    generateData();
    parametersChanged = true;

    console.log('INFO: cleared  baseline train and test data sets');
  });

  // store data into the baseline data set
  d3.select("#data-store-button").on("click", () => {
    storeTrainAndTestData();
     console.log('INFO: stored current train and test data into baseline train and test data sets');
  });

  // store data into the baseline data set
  d3.select("#data-add-button").on("click", () => {
    // check whether this is the first data set stored
    if(baseline_trainData == null || baseline_testData == null || baseline_numSamples_train <=0){
      console.log('INFO: first stored data set');
      storeTrainAndTestData();
      return;
    }
    console.log('TEST: state.problem: ' + state.problem + ", baseline_problemType: " + baseline_problemType);

    // check whether the problem type is the same
     if(state.problem != baseline_problemType){
      console.log('ERROR: data sets from classification and regression problems cannot be added');
      return;
    }
     // TODO is the array dynamically allocated (so called lazy initialization)?
    // add the points
    let offset = baseline_trainData.length;
    for (let i = 0; i < trainData.length; i++) {
      baseline_trainData[offset + i] = trainData[i];
    }
    offset = baseline_testData.length;
    for (let i = 0; i < testData.length; i++) {
      baseline_testData[offset + i] = testData[i];
    }
    baseline_numSamples_train += trainData.length;
    baseline_numSamples_test += testData.length;
    console.log('INFO: added current train and test data into baseline train and test data sets');
  });
  // remove the last added data set from the baseline data set of points
  d3.select("#data-remove-button").on("click", () => {
    // check whether baseline data set is empty
    if(baseline_trainData == null || baseline_testData == null || baseline_numSamples_train <=0){
      console.log('ERROR: there is no stored data set');
      return;
    }
    console.log('TEST: state.problem: ' + state.problem + ", baseline_problemType: " + baseline_problemType);

    // check whether the problem type is the same
    if(state.problem != baseline_problemType){
      console.log('ERROR: data sets from classification and regression problems cannot be subtracted/mixed');
      return;
    }
    // check whether the train and test lengths of baseline is large enough
    if(trainData.length >= baseline_trainData.length || testData.length >= baseline_testData.length){
      console.log('ERROR: the baseline train data set has smaller number of points than the current dataset: train: ' +
          trainData.length + ", baseline_trainData.length: "  + baseline_trainData.length);
      console.log('ERROR: the baseline test data set has smaller number of points than the current dataset: train: ' +
          testData.length + ", baseline_testData.length: "  + baseline_trainData.length);
      return;
    }
    // remove the points using array.splice(start, deleteCount)
    let offset = baseline_trainData.length - trainData.length;
    baseline_trainData.splice(offset,trainData.length);

    offset = baseline_testData.length - testData.length;
    baseline_testData.splice(offset,testData.length);

    baseline_numSamples_train -= trainData.length;
    baseline_numSamples_test -= testData.length;
    console.log('INFO: added current train and test data into baseline train and test data sets');
  });

  // restore data from the baseline data set
  d3.select("#data-restore-button").on("click", () => {
    // check that a baseline model has been saved
    if(baseline_trainData == null || baseline_testData == null){
      console.log('ERROR: missing baseline train or test data');
      return;
    }
    setTrainAndTestData();
    parametersChanged = true;
    console.log('INFO: restored from memory baseline train and test data');
  });



  // compute network efficiency metrics and show histograms
  d3.select("#data-nn-klmetric-button").on("click", () => {
    // compute KL divergence metric reflecting the NN configurations (weights and biases) from training data points
    // compute the network efficiency per layer
    //let numSamples: number = (state.problem === Problem.REGRESSION) ? NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;
    let numSamples: number = current_numSamples_train + current_numSamples_test;
    /////////////////////////////////////////
    // evaluate training data
    let numEvalSamples: number = numSamples * state.percTrainData / 100;
     let success: boolean = netKLcoef.getNetworkInefficiencyPerLayer(network,trainData, numEvalSamples);
    if (!success){
      console.log('ERROR: KL divergence computation for train data failed');
      return;
    }
    let netEfficiency_N: number[] = netKLcoef.getNetEfficiency_N();
    let netEfficiency_P: number[] = netKLcoef.getNetEfficiency_P();

    // print the histograms and create histogram visualization
    let hist = new AppendingHistogramChart(netKLcoef.getMapGlobal(), netEfficiency_N, netEfficiency_P);
    let title: string = 'Train data: state histogram';
    let kl_metric_result: string = '&nbsp; TRAIN data <BR>' + hist.showKLHistogram('histDivTrain', title);

    kl_metric_result += '&nbsp; arithmetic avg KL value (N+P):' + (Math.round(netKLcoef.getArithmeticAvgKLdivergence() * 1000) / 1000).toString() + '<BR>';
    kl_metric_result += '&nbsp; geometric avg KL value (N+P):' + (Math.round(netKLcoef.getGeometricAvgKLdivergence() * 1000) / 1000).toString() + '<BR>';

    //  compute the extra information for the table with state info
    netKLcoef.computeLabelPredictionOverlap();
    /////////////////////////////////////////////////////////////////////////
    let caption: string = 'Train data: KL divergence info  <BR>';
    let mytable= new AppendingTableChart(netKLcoef);
    mytable.showTableKL('tableKLDivTrain', caption);
    caption = 'Train data: unique state info <BR>';
    mytable.showTableOverlap('tableOverlapDivTrain', caption);
    caption = 'Train data: all non-zero states <BR>';
    mytable.showTableStates('tableStatesDivTrain',caption);
    ////////////////////////////////////////////////////////////////////
/*
    // append the bin counts of states
    let count_states_result: string = '&nbsp; Bin Count of All States Assigned to Each Class Label Per Layer  <BR>';
    count_states_result += netKLcoef.convertStateBinCountToString();
    kl_metric_result += count_states_result;

    // append the most frequently utilized state by each class label in each layer
    let path_states_result: string = '&nbsp; Most frequently utilized state by each class label in each layer  <BR>';
    path_states_result += netKLcoef.convertStatePath('Max');
    kl_metric_result += path_states_result;

    // append the least frequently utilized state by each class label in each layer
    path_states_result = '&nbsp; Least frequently utilized state by each class label in each layer  <BR>';
    path_states_result += netKLcoef.convertStatePath('Min');
    kl_metric_result += path_states_result;
*/

    ///////////////////////////////////////////////////////////////////////
    // evaluate test data
    numEvalSamples = numSamples * (100 - state.percTrainData) / 100;
    netKLcoef.reset();
    success = netKLcoef.getNetworkInefficiencyPerLayer(network,testData, numEvalSamples);
    if (!success){
      console.log('ERROR: KL divergence computation for test data failed');
      return;
    }
    netEfficiency_N = netKLcoef.getNetEfficiency_N();
    netEfficiency_P = netKLcoef.getNetEfficiency_P();

    // print the histograms and create histogram visualization
    let histTest = new AppendingHistogramChart(netKLcoef.getMapGlobal(), netEfficiency_N, netEfficiency_P);
    title = 'Test data: state histogram';
    kl_metric_result += '&nbsp; TEST data <BR>' + histTest.showKLHistogram('histDivTest', title);

    kl_metric_result += '&nbsp; arithmetic avg KL value (N+P):' + (Math.round(netKLcoef.getArithmeticAvgKLdivergence() * 1000) / 1000).toString() + '<BR>';
    kl_metric_result += '&nbsp; geometric avg KL value (N+P):' + (Math.round(netKLcoef.getGeometricAvgKLdivergence() * 1000) / 1000).toString() + '<BR>';

    //  compute the extra information for the table with state info
    netKLcoef.computeLabelPredictionOverlap();
    /////////////////////////////////////////////////////////////////////////
    caption = 'Test data: KL divergence info <BR>';
    let mytableTest= new AppendingTableChart(netKLcoef);
    mytableTest.showTableKL('tableKLDivTest', caption);
    caption = 'Test data: unique state info <BR>';
    mytableTest.showTableOverlap('tableOverlapDivTest', caption);
    caption = 'Test data: all non-zero states <BR>';
    mytableTest.showTableStates('tableStatesDivTest',caption);
    ////////////////////////////////////////////////////////////////////

    /*
    // append the bin counts of states
    count_states_result  = '&nbsp; Bin Count of All States Assigned to Each Class Label Per Layer  <BR>';
    count_states_result += netKLcoef.convertStateBinCountToString();
    kl_metric_result += count_states_result;

    // append the most frequently utilized state by each class label in each layer
    path_states_result = '&nbsp; Most frequently utilized state by each class label in each layer  <BR>';
    path_states_result += netKLcoef.convertStatePath('Max');
    kl_metric_result += path_states_result;

    // append the least frequently utilized state by each class label in each layer
    path_states_result = '&nbsp; Least frequently utilized state by each class label in each layer  <BR>';
    path_states_result += netKLcoef.convertStatePath('Min');
    kl_metric_result += path_states_result;
*/

    let element = document.getElementById("KLdivergenceDiv");
    element.innerHTML = kl_metric_result;

  });

  // compute variation (average and stdev) of KL divergence over multiple runs (cross -validations)
  d3.select("#data-nn-xvalmetric-button").on("click", () => {

    //document.getElementById("data-nn-xvalmetric-button").style.cursor = "wait";

    let maxRuns: number = 3; // number of runs
    let max_epoch: number = 100; // number of epochs per run
    let idx: number;
    let xvalIdx: number;

    //let numSamples: number = (state.problem === Problem.REGRESSION) ? NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;
    let numSamples: number = current_numSamples_train + current_numSamples_test;

    let numEvalSamples: number = numSamples * state.percTrainData / 100;
    let netKLcoef = new AppendingNetworkEfficiency();
    let sum_N: number []  = []; // average sum
    let sum2_N: number [] = []; // stdev sum2
    let sum_P: number []  = []; // average sum
    let sum2_P: number [] = []; // stdev sum2

    // these loops go over the number of cross-validation runs (maxRuns)
    // and over the number of steps (or epochs of training in each cross-validation run)
    for (xvalIdx = 0; xvalIdx < maxRuns; xvalIdx++) {
      for (idx = 0; idx < max_epoch; idx++) {
        generateData();
        parametersChanged = true;

        //player.pause();
        userHasInteracted();
        if (iter === 0) {
          simulationStarted();
        }
        oneStep();
      }

      let success: boolean = netKLcoef.getNetworkInefficiencyPerLayer(network,trainData, numEvalSamples);
      if (!success){
        console.log('ERROR: KL divergence computation for train data failed in the run ID:' + xvalIdx);
        return;
      }
      let netEfficiency_N: number[] = netKLcoef.getNetEfficiency_N();
      let netEfficiency_P: number[] = netKLcoef.getNetEfficiency_P();
      // sanity check
      if(netEfficiency_N.length != netEfficiency_P.length){
        console.log('ERROR: unequal length of netEfficiency_N.length:' + netEfficiency_N.length + ', netEfficiency_P.length:'+netEfficiency_P.length);
        return;
      }
      if (xvalIdx == 0) {
        for (let i = 0; i < netEfficiency_N.length; i++) {
            sum_N[i] = 0.0;
            sum2_N[i] = 0.0;
            sum_P[i] = 0.0;
            sum2_P[i] = 0.0;
        }
      }
      for (let i = 0; i < netEfficiency_N.length; i++) {
        sum_N[i] += netEfficiency_N[i];
        sum2_N[i] += netEfficiency_N[i]*netEfficiency_N[i];
        sum_P[i] += netEfficiency_P[i];
        sum2_P[i] += netEfficiency_P[i]*netEfficiency_P[i];
      }

    }

    let kl_stats: string = '&nbsp; KL divergence stats over '+ maxRuns.toString() + ' cross-validation runs and max epochs ' + max_epoch.toString() + '<BR>';
    // compute average and stdev of each KL divergence value per layer
    for (let i = 0; i < sum_N.length; i++) {
      // stdev for N
      sum2_N[i] = sum2_N[i] - sum_N[i] * sum_N[i]/maxRuns;
      sum2_N[i] = Math.sqrt(sum2_N[i]/maxRuns);
      // avg for N
      sum_N[i] = sum_N[i]/maxRuns;
      // stdev for P
      sum2_P[i] = sum2_P[i] - sum_P[i] * sum_P[i]/maxRuns;
      sum2_P[i] = Math.sqrt(sum2_P[i]/maxRuns);
      // avg for P
      sum_P[i] = sum_P[i]/maxRuns;

      //console.log('layer:'+i+", avg KL(N):" + sum_N[i] + ', stdev KL(N):' + sum2_N[i]);
      //console.log('layer:'+i+", avg KL(P):" + sum_P[i] + ', stdev KL(P):' + sum2_P[i]);
      kl_stats += '&nbsp; layer:' + i.toString() + ', avg KL(N):' + (Math.round(sum_N[i]*1000)/1000).toString() + ', stdev KL(N):'
          + (Math.round(sum2_N[i]*1000)/1000).toString() + '<BR>';
      kl_stats += '&nbsp; layer:' + i.toString() + ', avg KL(P):' + (Math.round(sum_P[i]*1000)/1000).toString() + ', stdev KL(P):'
          + (Math.round(sum2_P[i]*1000)/1000).toString() + '<BR>';
    }

     let element = document.getElementById("KLdivergenceStatsDiv");
     element.innerHTML = kl_stats;

   // document.getElementById("data-nn-xvalmetric-button").style.cursor = "default";

  });

  // inference button with the current data set and nn model
  d3.select("#data-nn-infer-button").on("click", () => {
    // Compute the loss.
    lossTrain = getLoss(network, trainData);
    lossTest = getLoss(network, testData);
    //updateUI();
    let mse_result: string;
    mse_result = '&nbsp; MSE Train loss: ' + (Math.round(lossTrain * 1000) / 1000).toString() + ', ';
    mse_result += ' MSE Test loss: ' + (Math.round(lossTest * 1000) / 1000).toString() +  '<BR>';
    let element = document.getElementById("accuracyDiv");
    element.innerHTML =  mse_result;

    //console.log('TEST: MSE lossTrain=' + lossTrain + ', MSE lossTest=' + lossTest);
    //console.log('INFO: inference with the current data set and nn model');
  });

  // clear weights and biases for the baseline network
  d3.select("#nn-clear-button").on("click", () => {
    baseline_weights = null;
    baseline_biases = null;
    count_baseline_add = 0;
    count_baseline_subtract = 0;
    console.log('INFO: cleared baseline weights and biases');
  });
  // store weights and biases for the baseline network
  d3.select("#nn-storemmodel-button").on("click", () => {
    baseline_weights = null;
    baseline_biases = null;
    baseline_weights = getOutputWeights(network);
    baseline_biases = getOutputBiases(network);
    count_baseline_add = 1;
    count_baseline_subtract = 0;
    console.log('INFO: set memory to baseline weights and biases');
  });
  // restore weights and biases from the baseline network
  d3.select("#nn-restoremodel-button").on("click", () => {

    // check that a baseline model has been saved
    if(baseline_weights == null || baseline_biases == null){
      console.log('ERROR: missing baseline weights and biases');
      return;
    }
    // set the network to the baseline weights and biases
    if( !setOutputWeights(network, baseline_weights)){
      console.log('ERROR: failed to update weights');
    }
    if( !setOutputBiases(network, baseline_biases) ){
      console.log('ERROR: failed to update biases');
    }
    let firstStep = false;
    updateUI(firstStep = false);
    console.log('INFO: restored from memory all baseline weights and biases');
  });

  // subtract the current model weights and biases from baseline weights and biases and set the model
  d3.select("#nn-subtract-button").on("click", () => {

    // this is the case of subtracting a model from zero/empty baseline
    if(baseline_weights == null || baseline_biases == null){
      console.log('INFO: missing baseline weights and biases. They are assumed to be zeros!');
      baseline_weights = getOutputWeights(network);
      baseline_biases = getOutputBiases(network);
      for(let i = 0; i < baseline_weights.length; i++){
        baseline_weights[i] = 0 - baseline_weights[i];
        //console.log('new weight[' + (i) + ']:'  + baseline_weights[i]);
      }
      for(let j = 0; j < baseline_biases.length; j++){
        baseline_biases[j] = 0 - baseline_biases[j];
        //console.log('new bias[' + (j) + ']:' + baseline_biases[j]);
      }
      count_baseline_add = 0;
      count_baseline_subtract = 1;
      console.log('INFO: subtracted baseline weights and biases');
      return;
    }

    let weights: number[]; //Array<number>;
    weights = getOutputWeights(network);
    if(baseline_weights.length != weights.length){
      console.log('ERROR: baseline network architecture is different from the current architecture');
      console.log('number of baseline weights:' + baseline_weights.length + ', number of current weights: ' +weights.length);
      return;
    }

    for(let i = 0; i < weights.length; i++){
      baseline_weights[i] = weights[i] - baseline_weights[i];
      //console.log('delta weight[' + (i) + ']:'  + weights[i]);
    }

    let biases: number[]; //Array<number>;
    biases = getOutputBiases(network);
    if(baseline_biases.length != biases.length){
      console.log('ERROR: baseline network architecture is different from the current architecture');
      console.log('number of baseline biases:' + baseline_biases.length + ', number of current biases: ' +biases.length);
      return;
    }
    for(let j = 0; j < biases.length; j++){
      baseline_biases[j] = biases[j] - baseline_biases[j];
      //console.log('delta bias[' + (j) + ']:' + biases[j]);
    }

    count_baseline_subtract ++;
    let firstStep = false;
    updateUI(firstStep = false);
    console.log('INFO: subtracted baseline weights and biases from current weights and biases');
  });

  // add the current model weights and biases and the baseline weights and biases
  d3.select("#nn-add-button").on("click", () => {

    // this is the case of adding a model to zero/empty baseline model
    if(baseline_weights == null || baseline_biases == null){
      console.log('INFO: missing baseline weights and biases. They are assumed to be zeros!');
      baseline_weights = getOutputWeights(network);
      baseline_biases = getOutputBiases(network);
      count_baseline_add = 1;
      count_baseline_subtract = 0;
      console.log('INFO: added/set baseline weights and biases');
      return;
    }
    let weights: number[]; //Array<number>;
    weights = getOutputWeights(network);
    if(baseline_weights.length != weights.length){
      console.log('ERROR: baseline network architecture is different from the current architecture');
      console.log('number of baseline weights:' + baseline_weights.length + ', number of current weights: ' +weights.length);
      return;
    }

    for(let i = 0; i < weights.length; i++){
      baseline_weights[i] = weights[i] + baseline_weights[i];
      //onsole.log('new weight[' + (i) + ']:'  + weights[i]);
    }

    let biases: number[]; //Array<number>;
    biases = getOutputBiases(network);
    if(baseline_biases.length != biases.length){
      console.log('ERROR: baseline network architecture is different from the current architecture');
      console.log('number of baseline biases:' + baseline_biases.length + ', number of current biases: ' +biases.length);
      return;
    }
    for(let j = 0; j < biases.length; j++){
      baseline_biases[j] = biases[j] + baseline_biases[j];
      //console.log('new bias[' + (j) + ']:' + biases[j]);
    }

    count_baseline_add ++;
    let firstStep = false;
    updateUI(firstStep = false);
    console.log('INFO: added current weights and biases to baseline weights and biases');
    //writeNetwork(network);
  });
  // average all baseline weights and biases based on the number of added models
  d3.select("#nn-avg-button").on("click", () => {

    if(baseline_weights == null || baseline_biases == null){
      console.log('ERROR: missing baseline weights and biases');
      return;
    }
    if(count_baseline_add < 2){
      console.log('INFO: there is only one model (no averaging): count_baseline_add =' + count_baseline_add.toString());
      return;
    }

    for(let i = 0; i < baseline_weights.length; i++){
      baseline_weights[i] =  baseline_weights[i]/count_baseline_add;
      console.log('avg weight[' + (i) + ']:'  + baseline_weights[i]);
    }

    for(let j = 0; j < baseline_biases.length; j++){
      baseline_biases[j] = baseline_biases[j]/count_baseline_add;
      console.log('avg bias[' + (j) + ']:' + baseline_biases[j]);
    }

    console.log('INFO: averaged count_baseline_add =' + count_baseline_add.toString() + ' models stored in memory');
    count_baseline_add = 1; // reset the count
    let firstStep = false;
    updateUI(firstStep = false);

  });

  // save the current model to a CSV file
  // links of interest
  // https://github.com/microsoft/onnxjs
  // resnet50 demo using ONNX.js - https://microsoft.github.io/onnxjs-demo/#/resnet50
  // we need a write in JavaScript - https://github.com/onnx/tutorials
  d3.select("#save-nn-button").on("click", () => {
    let myio= new AppendingInputOutput();
    myio.writeNetwork(network);
  });
  d3.select("#save-statehist-button").on("click", () => {
    let myio= new AppendingInputOutput();
    myio.writeStateHist(netKLcoef);
    myio.writeKLdivergence(netKLcoef);
    myio.writeOverlapStates(netKLcoef);
  });


  let dataThumbnails = d3.selectAll("canvas[data-dataset]");
  dataThumbnails.on("click", function() {
    let newDataset = datasets[this.dataset.dataset];
    if (newDataset === state.dataset) {
      return; // No-op.
    }
    state.dataset =  newDataset;
    dataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;
    reset();
  });

  let datasetKey = getKeyFromValue(datasets, state.dataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-dataset=${datasetKey}]`)
    .classed("selected", true);

  let regDataThumbnails = d3.selectAll("canvas[data-regDataset]");
  regDataThumbnails.on("click", function() {
    let newDataset = regDatasets[this.dataset.regdataset];
    if (newDataset === state.regDataset) {
      return; // No-op.
    }
    state.regDataset =  newDataset;
    regDataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;
    reset();
  });

  let regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
    .classed("selected", true);

  d3.select("#add-layers").on("click", () => {
    if (state.numHiddenLayers >= 6) {
      return;
    }
    state.networkShape[state.numHiddenLayers] = 2;
    state.numHiddenLayers++;
    parametersChanged = true;
    reset();
  });

  d3.select("#remove-layers").on("click", () => {
    if (state.numHiddenLayers <= 0) {
      return;
    }
    state.numHiddenLayers--;
    state.networkShape.splice(state.numHiddenLayers);
    parametersChanged = true;
    reset();
  });

  let showTestData = d3.select("#show-test-data").on("change", function() {
    state.showTestData = this.checked;
    state.serialize();
    userHasInteracted();
    heatMap.updateTestPoints(state.showTestData ? testData : []);
  });
  // Check/uncheck the checkbox according to the current state.
  showTestData.property("checked", state.showTestData);

  let discretize = d3.select("#discretize").on("change", function() {
    state.discretize = this.checked;
    state.serialize();
    userHasInteracted();
    updateUI();
  });
  // Check/uncheck the checbox according to the current state.
  discretize.property("checked", state.discretize);

  let percTrain = d3.select("#percTrainData").on("input", function() {
    state.percTrainData = this.value;
    d3.select("label[for='percTrainData'] .value").text(this.value);
    generateData();
    parametersChanged = true;
    reset();
  });
  percTrain.property("value", state.percTrainData);
  d3.select("label[for='percTrainData'] .value").text(state.percTrainData);

  let noise = d3.select("#noise").on("input", function() {
    state.noise = this.value;
    d3.select("label[for='noise'] .value").text(this.value);
    generateData();
    parametersChanged = true;
    reset();
  });
  let currentMax = parseInt(noise.property("max"));
  if (state.noise > currentMax) {
    if (state.noise <= 80) {
      noise.property("max", state.noise);
    } else {
      state.noise = 50;
    }
  } else if (state.noise < 0) {
    state.noise = 0;
  }
  noise.property("value", state.noise);
  d3.select("label[for='noise'] .value").text(state.noise);

	// added trojan = number of randomly selected points switching labels
  let trojan = d3.select("#trojan").on("input", function() {
    state.trojan = this.value;
    d3.select("label[for='trojan'] .value").text(this.value);
	// to swap randomly labels
    swapDataLabels();
    parametersChanged = true;
    reset();
  });
  let currentTrojanMax = parseInt(trojan.property("max"));
  if (state.trojan > currentTrojanMax) {
    if (state.trojan <= 8) {
      trojan.property("max", state.trojan);
    } else {
      state.trojan = 1;
    }
  } else if (state.trojan < 0) {
    state.trojan = 0;
  }
  trojan.property("value", state.trojan);
  d3.select("label[for='trojan'] .value").text(state.trojan);
  
  let batchSize = d3.select("#batchSize").on("input", function() {
    state.batchSize = this.value;
    d3.select("label[for='batchSize'] .value").text(this.value);
    parametersChanged = true;
    reset();
  });
  batchSize.property("value", state.batchSize);
  d3.select("label[for='batchSize'] .value").text(state.batchSize);

  let activationDropdown = d3.select("#activations").on("change", function() {
    state.activation = activations[this.value];
    parametersChanged = true;
    reset();
  });
  activationDropdown.property("value",
      getKeyFromValue(activations, state.activation));

  let learningRate = d3.select("#learningRate").on("change", function() {
    state.learningRate = +this.value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
  });
  learningRate.property("value", state.learningRate);

  let regularDropdown = d3.select("#regularizations").on("change",
      function() {
    state.regularization = regularizations[this.value];
    parametersChanged = true;
    reset();
  });
  regularDropdown.property("value",
      getKeyFromValue(regularizations, state.regularization));

  let regularRate = d3.select("#regularRate").on("change", function() {
    state.regularizationRate = +this.value;
    parametersChanged = true;
    reset();
  });
  regularRate.property("value", state.regularizationRate);

  let problem = d3.select("#problem").on("change", function() {
    state.problem = problems[this.value];
    generateData();
    drawDatasetThumbnails();
    parametersChanged = true;
    reset();
  });
  problem.property("value", getKeyFromValue(problems, state.problem));

  // Add scale to the gradient color map.
  let x = d3.scale.linear().domain([-1, 1]).range([0, 144]);
  let xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .tickValues([-1, 0, 1])
    .tickFormat(d3.format("d"));
  d3.select("#colormap g.core").append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0,10)")
    .call(xAxis);

  // Listen for css-responsive changes and redraw the svg network.

  window.addEventListener("resize", () => {
    let newWidth = document.querySelector("#main-part")
        .getBoundingClientRect().width;
    if (newWidth !== mainWidth) {
      mainWidth = newWidth;
      drawNetwork(network);
      updateUI(true);
    }
  });

  // Hide the text below the visualization depending on the URL.
  if (state.hideText) {
    d3.select("#article-text").style("display", "none");
    d3.select("div.more").style("display", "none");
    d3.select("header").style("display", "none");
  }
}

function updateBiasesUI(network: nn.Node[][]) {
  nn.forEachNode(network, true, node => {
    d3.select(`rect#bias-${node.id}`).style("fill", colorScale(node.bias));
  });
}

function updateWeightsUI(network: nn.Node[][], container) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        container.select(`#link${link.source.id}-${link.dest.id}`)
            .style({
              "stroke-dashoffset": -iter / 3,
              "stroke-width": linkWidthScale(Math.abs(link.weight)),
              "stroke": colorScale(link.weight)
            })
            .datum(link);
      }
    }
  }
}

function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
    container, node?: nn.Node) {
  let x = cx - RECT_SIZE / 2;
  let y = cy - RECT_SIZE / 2;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "transform": `translate(${x},${y})`
    });

  // Draw the main rectangle.
  nodeGroup.append("rect")
    .attr({
      x: 0,
      y: 0,
      width: RECT_SIZE,
      height: RECT_SIZE,
    });
  let activeOrNotClass = state[nodeId] ? "active" : "inactive";
  if (isInput) {
    let label = INPUTS[nodeId].label != null ?
        INPUTS[nodeId].label : nodeId;
    // Draw the input label.
    let text = nodeGroup.append("text").attr({
      class: "main-label",
      x: -10,
      y: RECT_SIZE / 2, "text-anchor": "end"
    });
    if (/[_^]/.test(label)) {
      let myRe = /(.*?)([_^])(.)/g;
      let myArray;
      let lastIndex;
      while ((myArray = myRe.exec(label)) != null) {
        lastIndex = myRe.lastIndex;
        let prefix = myArray[1];
        let sep = myArray[2];
        let suffix = myArray[3];
        if (prefix) {
          text.append("tspan").text(prefix);
        }
        text.append("tspan")
        .attr("baseline-shift", sep === "_" ? "sub" : "super")
        .style("font-size", "9px")
        .text(suffix);
      }
      if (label.substring(lastIndex)) {
        text.append("tspan").text(label.substring(lastIndex));
      }
    } else {
      text.append("tspan").text(label);
    }
    nodeGroup.classed(activeOrNotClass, true);
  }
  if (!isInput) {
    // Draw the node's bias.
    nodeGroup.append("rect")
      .attr({
        id: `bias-${nodeId}`,
        x: -BIAS_SIZE - 2,
        y: RECT_SIZE - BIAS_SIZE + 3,
        width: BIAS_SIZE,
        height: BIAS_SIZE,
      }).on("mouseenter", function() {
        updateHoverCard(HoverType.BIAS, node, d3.mouse(container.node()));
      }).on("mouseleave", function() {
        updateHoverCard(null);
      });
  }

  // Draw the node's canvas.
  let div = d3.select("#network").insert("div", ":first-child")
    .attr({
      "id": `canvas-${nodeId}`,
      "class": "canvas"
    })
    .style({
      position: "absolute",
      left: `${x + 3}px`,
      top: `${y + 3}px`
    })
    .on("mouseenter", function() {
      selectedNodeId = nodeId;
      div.classed("hovered", true);
      nodeGroup.classed("hovered", true);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[nodeId], state.discretize);
    })
    .on("mouseleave", function() {
      selectedNodeId = null;
      div.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[nn.getOutputNode(network).id],
          state.discretize);
    });
  if (isInput) {
    div.on("click", function() {
      state[nodeId] = !state[nodeId];
      parametersChanged = true;
      reset();
    });
    div.style("cursor", "pointer");
  }
  if (isInput) {
    div.classed(activeOrNotClass, true);
  }
  let nodeHeatMap = new HeatMap(RECT_SIZE, DENSITY / 10, xDomain,
      xDomain, div, {noSvg: true});
  div.datum({heatmap: nodeHeatMap, id: nodeId});

}

// Draw network
function drawNetwork(network: nn.Node[][]): void {
  let svg = d3.select("#svg");
  // Remove all svg elements.
  svg.select("g.core").remove();
  // Remove all div elements.
  d3.select("#network").selectAll("div.canvas").remove();
  d3.select("#network").selectAll("div.plus-minus-neurons").remove();

  // Get the width of the svg container.
  let padding = 3;
  let co = d3.select(".column.output").node() as HTMLDivElement;
  let cf = d3.select(".column.features").node() as HTMLDivElement;
  let width = co.offsetLeft - cf.offsetLeft;
  svg.attr("width", width);

  // Map of all node coordinates.
  let node2coord: {[id: string]: {cx: number, cy: number}} = {};
  let container = svg.append("g")
    .classed("core", true)
    .attr("transform", `translate(${padding},${padding})`);
  // Draw the network layer by layer.
  let numLayers = network.length;
  let featureWidth = 118;
  let layerScale = d3.scale.ordinal<number, number>()
      .domain(d3.range(1, numLayers - 1))
      .rangePoints([featureWidth, width - RECT_SIZE], 0.7);
  let nodeIndexScale = (nodeIndex: number) => nodeIndex * (RECT_SIZE + 25);


  let calloutThumb = d3.select(".callout.thumbnail").style("display", "none");
  let calloutWeights = d3.select(".callout.weights").style("display", "none");
  let idWithCallout = null;
  let targetIdWithCallout = null;

  // Draw the input layer separately.
  let cx = RECT_SIZE / 2 + 50;
  let nodeIds = Object.keys(INPUTS);
  let maxY = nodeIndexScale(nodeIds.length);
  nodeIds.forEach((nodeId, i) => {
    let cy = nodeIndexScale(i) + RECT_SIZE / 2;
    node2coord[nodeId] = {cx, cy};
    drawNode(cx, cy, nodeId, true, container);
  });

  // Draw the intermediate layers.
  for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
    let numNodes = network[layerIdx].length;
    let cx = layerScale(layerIdx) + RECT_SIZE / 2;
    maxY = Math.max(maxY, nodeIndexScale(numNodes));
    addPlusMinusControl(layerScale(layerIdx), layerIdx);
    for (let i = 0; i < numNodes; i++) {
      let node = network[layerIdx][i];
      let cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[node.id] = {cx, cy};
      drawNode(cx, cy, node.id, false, container, node);

      // Show callout to thumbnails.
      let numNodes = network[layerIdx].length;
      let nextNumNodes = network[layerIdx + 1].length;
      if (idWithCallout == null &&
          i === numNodes - 1 &&
          nextNumNodes <= numNodes) {
        calloutThumb.style({
          display: null,
          top: `${20 + 3 + cy}px`,
          left: `${cx}px`
        });
        idWithCallout = node.id;
      }

      // Draw links.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        let path: SVGPathElement = drawLink(link, node2coord, network,
            container, j === 0, j, node.inputLinks.length).node() as any;
        // Show callout to weights.
        let prevLayer = network[layerIdx - 1];
        let lastNodePrevLayer = prevLayer[prevLayer.length - 1];
        if (targetIdWithCallout == null &&
            i === numNodes - 1 &&
            link.source.id === lastNodePrevLayer.id &&
            (link.source.id !== idWithCallout || numLayers <= 5) &&
            link.dest.id !== idWithCallout &&
            prevLayer.length >= numNodes) {
          let midPoint = path.getPointAtLength(path.getTotalLength() * 0.7);
          calloutWeights.style({
            display: null,
            top: `${midPoint.y + 5}px`,
            left: `${midPoint.x + 3}px`
          });
          targetIdWithCallout = link.dest.id;
        }
      }
    }
  }

  // Draw the output node separately.
  cx = width + RECT_SIZE / 2;
  let node = network[numLayers - 1][0];
  let cy = nodeIndexScale(0) + RECT_SIZE / 2;
  node2coord[node.id] = {cx, cy};
  // Draw links.
  for (let i = 0; i < node.inputLinks.length; i++) {
    let link = node.inputLinks[i];
    drawLink(link, node2coord, network, container, i === 0, i,
        node.inputLinks.length);
  }
  // Adjust the height of the svg.
  svg.attr("height", maxY);

  // Adjust the height of the features column.
  let height = Math.max(
    getRelativeHeight(calloutThumb),
    getRelativeHeight(calloutWeights),
    getRelativeHeight(d3.select("#network"))
  );
  d3.select(".column.features").style("height", height + "px");
}

function getRelativeHeight(selection) {
  let node = selection.node() as HTMLAnchorElement;
  return node.offsetHeight + node.offsetTop;
}

function addPlusMinusControl(x: number, layerIdx: number) {
  let div = d3.select("#network").append("div")
    .classed("plus-minus-neurons", true)
    .style("left", `${x - 10}px`);

  let i = layerIdx - 1;
  let firstRow = div.append("div").attr("class", `ui-numNodes${layerIdx}`);
  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons >= 8) {
          return;
        }
        state.networkShape[i]++;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("add");

  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons <= 1) {
          return;
        }
        state.networkShape[i]--;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("remove");

  let suffix = state.networkShape[i] > 1 ? "s" : "";
  div.append("div").text(
    state.networkShape[i] + " neuron" + suffix
  );
}

function updateHoverCard(type: HoverType, nodeOrLink?: nn.Node | nn.Link,
    coordinates?: [number, number]) {
  let hovercard = d3.select("#hovercard");
  if (type == null) {
    hovercard.style("display", "none");
    d3.select("#svg").on("click", null);
    return;
  }
  d3.select("#svg").on("click", () => {
    hovercard.select(".value").style("display", "none");
    let input = hovercard.select("input");
    input.style("display", null);
    input.on("input", function() {
      if (this.value != null && this.value !== "") {
        if (type === HoverType.WEIGHT) {
          (nodeOrLink as nn.Link).weight = +this.value;
        } else {
          (nodeOrLink as nn.Node).bias = +this.value;
        }
        updateUI();
      }
    });
    input.on("keypress", () => {
      if ((d3.event as any).keyCode === 13) {
        updateHoverCard(type, nodeOrLink, coordinates);
      }
    });
    (input.node() as HTMLInputElement).focus();
  });
  let value = (type === HoverType.WEIGHT) ?
    (nodeOrLink as nn.Link).weight :
    (nodeOrLink as nn.Node).bias;
  let name = (type === HoverType.WEIGHT) ? "Weight" : "Bias";
  hovercard.style({
    "left": `${coordinates[0] + 20}px`,
    "top": `${coordinates[1]}px`,
    "display": "block"
  });
  hovercard.select(".type").text(name);
  hovercard.select(".value")
    .style("display", null)
    .text(value.toPrecision(2));
  hovercard.select("input")
    .property("value", value.toPrecision(2))
    .style("display", "none");
}

function drawLink(
    input: nn.Link, node2coord: {[id: string]: {cx: number, cy: number}},
    network: nn.Node[][], container,
    isFirst: boolean, index: number, length: number) {
  let line = container.insert("path", ":first-child");
  let source = node2coord[input.source.id];
  let dest = node2coord[input.dest.id];
  let datum = {
    source: {
      y: source.cx + RECT_SIZE / 2 + 2,
      x: source.cy
    },
    target: {
      y: dest.cx - RECT_SIZE / 2,
      x: dest.cy + ((index - (length - 1) / 2) / length) * 12
    }
  };
  let diagonal = d3.svg.diagonal().projection(d => [d.y, d.x]);
  line.attr({
    "marker-start": "url(#markerArrow)",
    class: "link",
    id: "link" + input.source.id + "-" + input.dest.id,
    d: diagonal(datum, 0)
  });

  // Add an invisible thick link that will be used for
  // showing the weight value on hover.
  container.append("path")
    .attr("d", diagonal(datum, 0))
    .attr("class", "link-hover")
    .on("mouseenter", function() {
      updateHoverCard(HoverType.WEIGHT, input, d3.mouse(this));
    }).on("mouseleave", function() {
      updateHoverCard(null);
    });
  return line;
}

/**
 * Given a neural network, it asks the network for the output (prediction)
 * of every node in the network using inputs sampled on a square grid.
 * It returns a map where each key is the node ID and the value is a square
 * matrix of the outputs of the network for each input in the grid respectively.
 */
function updateDecisionBoundary(network: nn.Node[][], firstTime: boolean) {
  if (firstTime) {
    boundary = {};
    nn.forEachNode(network, true, node => {
      boundary[node.id] = new Array(DENSITY);
    });
    // Go through all predefined inputs.
    for (let nodeId in INPUTS) {
      boundary[nodeId] = new Array(DENSITY);
    }
  }
  let xScale = d3.scale.linear().domain([0, DENSITY - 1]).range(xDomain);
  let yScale = d3.scale.linear().domain([DENSITY - 1, 0]).range(xDomain);

  let i = 0, j = 0;
  for (i = 0; i < DENSITY; i++) {
    if (firstTime) {
      nn.forEachNode(network, true, node => {
        boundary[node.id][i] = new Array(DENSITY);
      });
      // Go through all predefined inputs.
      for (let nodeId in INPUTS) {
        boundary[nodeId][i] = new Array(DENSITY);
      }
    }
    for (j = 0; j < DENSITY; j++) {
      // 1 for points inside the circle, and 0 for points outside the circle.
      let x = xScale(i);
      let y = yScale(j);
      let input = constructInput(x, y);
      nn.forwardProp(network, input);
      nn.forEachNode(network, true, node => {
        boundary[node.id][i][j] = node.output;
      });
      if (firstTime) {
        // Go through all predefined inputs.
        for (let nodeId in INPUTS) {
          boundary[nodeId][i][j] = INPUTS[nodeId].f(x, y);
        }
      }
    }
  }
}

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let dataPoint = dataPoints[i];
    let input = constructInput(dataPoint.x, dataPoint.y);
    let output = nn.forwardProp(network, input);
    loss += nn.Errors.SQUARE.error(output, dataPoint.label);
  }
  return loss / dataPoints.length;
}

function updateUI(firstStep = false) {
  // Update the links visually.
  updateWeightsUI(network, d3.select("g.core"));
  // Update the bias values visually.
  updateBiasesUI(network);
  // Get the decision boundary of the network.
  updateDecisionBoundary(network, firstStep);
  let selectedId = selectedNodeId != null ?
      selectedNodeId : nn.getOutputNode(network).id;
  heatMap.updateBackground(boundary[selectedId], state.discretize);

  // Update all decision boundaries.
  d3.select("#network").selectAll("div.canvas")
      .each(function(data: {heatmap: HeatMap, id: string}) {
    data.heatmap.updateBackground(reduceMatrix(boundary[data.id], 10),
        state.discretize);
  });

  function zeroPad(n: number): string {
    let pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function humanReadable(n: number): string {
    return n.toFixed(3);
  }

  // Update loss and iteration number.
  d3.select("#loss-train").text(humanReadable(lossTrain));
  d3.select("#loss-test").text(humanReadable(lossTest));
  d3.select("#iter-number").text(addCommas(zeroPad(iter)));
  lineChart.addDataPoint([lossTrain, lossTest]);
}

function constructInputIds(): string[] {
  let result: string[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      result.push(inputName);
    }
  }
  return result;
}

export function constructInput(x: number, y: number): number[] {
  let input: number[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      input.push(INPUTS[inputName].f(x, y));
    }
  }
  return input;
}

function oneStep(): void {
  iter++;
  trainData.forEach((point, i) => {
    let input = constructInput(point.x, point.y);
    nn.forwardProp(network, input);
    nn.backProp(network, point.label, nn.Errors.SQUARE);
    if ((i + 1) % state.batchSize === 0) {
      nn.updateWeights(network, state.learningRate, state.regularizationRate);
    }
  });
  // Compute the loss.
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  updateUI();
}

// get network weights
export function getOutputWeights(network: nn.Node[][]): number[] {
  let weights: number[] = [];
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    let minLayer = 10.0;
    let maxLayer = -10.0;
    let closeToZero = 10.0;
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        weights.push(output.weight);
        /*
        // added to compute min and max - TODO simplify the code
        if (minLayer > output.weight) {
          minLayer = output.weight;
        }
        if (maxLayer < output.weight) {
          maxLayer = output.weight;
        }
        if (closeToZero > Math.abs(output.weight)) {
          closeToZero = Math.abs(output.weight);
        }
        */

      }
    }
    //console.log("layer:" + layerIdx + " minWWeight:" + minLayer + " maxWeight:" + maxLayer + " closeToZero:" + closeToZero);

    /*
    // TODO add the computation of average sparsity per node
    let sparsityLayer = 0.0;
    let number_of_links = 0;
    let maxAbsWeight = 0.0;
    if (Math.abs(minLayer) > Math.abs(maxLayer)) {
      maxAbsWeight = Math.abs(minLayer);
    } else {
      maxAbsWeight = Math.abs(maxLayer);
    }
    let percentWeightThresh = 0.1 * maxAbsWeight;
    console.log("layer:" + layerIdx + " maxAbsWWeight:" + maxAbsWeight + " percentWeightThresh:" + percentWeightThresh );

    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      let sparsityNode = 0.0;
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        if (Math.abs(output.weight) < percentWeightThresh) {
          sparsityNode = sparsityNode + 1;
        }
      }
      sparsityLayer = sparsityLayer + sparsityNode;
      number_of_links = number_of_links + node.outputs.length;
      // sparsity per node = percent of weights leaving the layer
      // that are less than 10 % of the abs max weight value per layer
      // this implies that a node should be removed/pruned if the value approaches one
      sparsityNode = sparsityNode / node.outputs.length;
      console.log("node:" + i + " sparsityNode:"+sparsityNode);
    }
    // this is the sparsity per layer = percent of weight leaving all nodes
    // that are less than 10 % of the abs max weight value per layer
    // this implies that the layer (i.e., a set of nodes) is not efficiently utilized
    sparsityLayer = sparsityLayer/number_of_links;
    console.log("layer:" + layerIdx + " sparsityLayer:"+sparsityLayer);
    */
  }
  return weights;
}

// set network weights
export function setOutputWeights(network: nn.Node[][], weights: number[]): boolean {
  //let weights: number[] = [];
  let idx = 0;
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.outputs.length; j++) {
        if(idx< weights.length) {
          node.outputs[j].weight = weights[idx];
          idx = idx + 1;
        }else{
          console.log("ERROR: mismatch of baseline and current network weights");
          return false;
        }
      }
    }
  }
  return true;
}

// get the current network biases
export function getOutputBiases(network: nn.Node[][]): number[] {
  let biases: number[] = [];
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      let output = node.bias;
        biases.push(output);
    }
  }
  return biases;
}
// set the network biases
export function setOutputBiases(network: nn.Node[][], biases: number[]): boolean {
  //let biases: number[] = [];
  let idx = 0;
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      if(idx<biases.length) {
        node.bias = biases[idx];
        idx = idx + 1;
      }else{
        console.log("ERROR: mismatch in baseline and current network biases");
        return false;
      }
    }
  }
  return true;
}

function reset(onStartup=false) {
  lineChart.reset();
  state.serialize();
  if (!onStartup) {
    userHasInteracted();
  }
  player.pause();

  let suffix = state.numHiddenLayers !== 1 ? "s" : "";
  d3.select("#layers-label").text("Hidden layer" + suffix);
  d3.select("#num-layers").text(state.numHiddenLayers);

  // Make a simple network.
  iter = 0;
  let numInputs = constructInput(0 , 0).length;
  let shape = [numInputs].concat(state.networkShape).concat([1]);
  let outputActivation = (state.problem === Problem.REGRESSION) ?
      nn.Activations.LINEAR : nn.Activations.TANH;
  network = nn.buildNetwork(shape, state.activation, outputActivation,
      state.regularization, constructInputIds(), state.initZero);
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  drawNetwork(network);
  updateUI(true);
};

function initTutorial() {
  if (state.tutorial == null || state.tutorial === '' || state.hideText) {
    return;
  }
  // Remove all other text.
  d3.selectAll("article div.l--body").remove();
  let tutorial = d3.select("article").append("div")
    .attr("class", "l--body");
  // Insert tutorial text.
  d3.html(`tutorials/${state.tutorial}.html`, (err, htmlFragment) => {
    if (err) {
      throw err;
    }
    tutorial.node().appendChild(htmlFragment);
    // If the tutorial has a <title> tag, set the page title to that.
    let title = tutorial.select("title");
    if (title.size()) {
      d3.select("header h1").style({
        "margin-top": "20px",
        "margin-bottom": "20px",
      })
      .text(title.text());
      document.title = title.text();
    }
  });
}

function drawDatasetThumbnails() {
  function renderThumbnail(canvas, dataGenerator) {
    let w = 100;
    let h = 100;
    canvas.setAttribute("width", w);
    canvas.setAttribute("height", h);
    let context = canvas.getContext("2d");
    let data = dataGenerator(200, 0);
    data.forEach(function(d) {
      context.fillStyle = colorScale(d.label);
      context.fillRect(w * (d.x + 6) / 12, h * (d.y + 6) / 12, 4, 4);
    });
    d3.select(canvas.parentNode).style("display", null);
  }
  d3.selectAll(".dataset").style("display", "none");

  if (state.problem === Problem.CLASSIFICATION) {
    for (let dataset in datasets) {
      let canvas: any =
          document.querySelector(`canvas[data-dataset=${dataset}]`);
      let dataGenerator = datasets[dataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
  if (state.problem === Problem.REGRESSION) {
    for (let regDataset in regDatasets) {
      let canvas: any =
          document.querySelector(`canvas[data-regDataset=${regDataset}]`);
      let dataGenerator = regDatasets[regDataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
}

function hideControls() {
  // Set display:none to all the UI elements that are hidden.
  let hiddenProps = state.getHiddenProps();
  hiddenProps.forEach(prop => {
    let controls = d3.selectAll(`.ui-${prop}`);
    if (controls.size() === 0) {
      console.warn(`0 html elements found with class .ui-${prop}`);
    }
    controls.style("display", "none");
  });

  // Also add checkbox for each hidable control in the "use it in classrom"
  // section.
  let hideControls = d3.select(".hide-controls");
  HIDABLE_CONTROLS.forEach(([text, id]) => {
    let label = hideControls.append("label")
      .attr("class", "mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect");
    let input = label.append("input")
      .attr({
        type: "checkbox",
        class: "mdl-checkbox__input",
      });
    if (hiddenProps.indexOf(id) === -1) {
      input.attr("checked", "true");
    }
    input.on("change", function() {
      state.setHideProperty(id, !this.checked);
      state.serialize();
      userHasInteracted();
      d3.select(".hide-controls-link")
        .attr("href", window.location.href);
    });
    label.append("span")
      .attr("class", "mdl-checkbox__label label")
      .text(text);
  });
  d3.select(".hide-controls-link")
    .attr("href", window.location.href);
}

function generateData(firstTime = false) {
  if (!firstTime) {
    // Change the seed.
    state.seed = Math.random().toFixed(5);
    state.serialize();
    userHasInteracted();
  }
  Math.seedrandom(state.seed);
  let numSamples = (state.problem === Problem.REGRESSION) ? NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;

  let generator = state.problem === Problem.CLASSIFICATION ?
      state.dataset : state.regDataset;
  let data = generator(numSamples, state.noise / 100, state.trojan);
  // Shuffle the data in-place.
  shuffle(data);
  // Split into train and test data.
  let splitIndex = Math.floor(data.length * state.percTrainData / 100);
  trainData = data.slice(0, splitIndex);
  testData = data.slice(splitIndex);
  // added to support the data calculator
  current_numSamples_train = trainData.length;
  current_numSamples_test = testData.length;
  heatMap.updatePoints(trainData);
  heatMap.updateTestPoints(state.showTestData ? testData : []);
}

// this function is used by the data part of the calculator
function setTrainAndTestData() {
  if (baseline_numSamples_train <= 0) {
    console.log("ERROR: the baseline training data set is empty. NumSamples = " + baseline_numSamples_train);
    return;
  }
  //console.log("INFO: setTrainAndTestData: the baseline training data NumSamples = " + baseline_numSamples_train + ", problem type " + baseline_problemType);
  // set the problem type
  state.problem = (baseline_problemType === Problem.REGRESSION) ? Problem.REGRESSION : Problem.CLASSIFICATION;
  //console.log("INFO: state.problem:: " + state.problem.toString());

  // update the drop-down menu for problem type
  let mySelect = <HTMLSelectElement>document.getElementById("problem");
  if (mySelect === null) {
    console.log("ERROR:missing problem type in index.html:" + mySelect);
  } else {
    console.log("current selection of problem: " + mySelect.selectedIndex); // 1
    // return the value of the selected option
    console.log("current value of problem type selection: " + mySelect.options[mySelect.selectedIndex].value) // Second

    let current_problemType = mySelect.selectedIndex;
    if (state.problem === Problem.CLASSIFICATION) {
      mySelect.selectedIndex = 0;
    } else {
      mySelect.selectedIndex = 1;
    }
    if(current_problemType != mySelect.selectedIndex ){
      // refresh the dataset icons
      drawDatasetThumbnails();
      console.log("refreshed data sets because of change of problem type");
    }
    parametersChanged = true;
    reset();
  }
  // set the number of data points
  current_numSamples_train = baseline_trainData.length;
  current_numSamples_test = baseline_testData.length;

  // copy the data points
  for (let i = 0; i < baseline_trainData.length; i++) {
     trainData[i] = baseline_trainData[i];
  }
  for (let i = 0; i < baseline_testData.length; i++) {
     testData[i] = baseline_testData[i];
  }

  //drawDatasetThumbnails();
  parametersChanged = true;
  reset();

  heatMap.updatePoints(trainData);
  heatMap.updateTestPoints(state.showTestData ? testData : []);

}

// this function is used by the data calculator
function storeTrainAndTestData(){

  if(trainData == null || testData == null || trainData.length < 1 || testData.length < 1){
    console.log("ERROR: trainData or testData is null or their length is less than 1");
    return;
  }
  //console.log("TEST: current state problemType: " + state.problem + ", current num sample in train:" + trainData.length);
  baseline_problemType = state.problem;
  baseline_numSamples_train = trainData.length;
  baseline_numSamples_test = testData.length;

  baseline_trainData = [];
  baseline_testData = [];

  // copy the data points
  for (let i = 0; i < trainData.length; i++) {
    baseline_trainData[i] = trainData[i];
  }
  for (let i = 0; i < testData.length; i++) {
    baseline_testData[i] = testData[i];
  }
}

function swapDataLabels(firstTime = false) {
  if (!firstTime) {
    // Change the seed.
    state.seed = Math.random().toFixed(5);
    state.serialize();
    userHasInteracted();
  }
  Math.seedrandom(state.seed);
  //let numSamples = (state.problem === Problem.REGRESSION) ? NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;
  let numSamples =   current_numSamples_train + current_numSamples_test;

  let generator = state.problem === Problem.CLASSIFICATION ?
      state.dataset : state.regDataset;
  let data = generator(numSamples, state.noise / 100, state.trojan );
  // Shuffle the data in-place.
  shuffle(data);
  // Split into train and test data.
  let splitIndex = Math.floor(data.length * state.percTrainData / 100);
  trainData = data.slice(0, splitIndex);
  testData = data.slice(splitIndex);
  heatMap.updatePoints(trainData);
  heatMap.updateTestPoints(state.showTestData ? testData : []);
}

let firstInteraction = true;
let parametersChanged = false;

function userHasInteracted() {
  if (!firstInteraction) {
    return;
  }
  firstInteraction = false;
  let page = 'index';
  if (state.tutorial != null && state.tutorial !== '') {
    page = `/v/tutorials/${state.tutorial}`;
  }
  ga('set', 'page', page);
  ga('send', 'pageview', {'sessionControl': 'start'});
}

function simulationStarted() {
  ga('send', {
    hitType: 'event',
    eventCategory: 'Starting Simulation',
    eventAction: parametersChanged ? 'changed' : 'unchanged',
    eventLabel: state.tutorial == null ? '' : state.tutorial
  });
  parametersChanged = false;
}


drawDatasetThumbnails();
initTutorial();
makeGUI();
generateData(true);
reset(true);
hideControls();

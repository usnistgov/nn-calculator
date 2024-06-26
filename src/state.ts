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

import * as nn from "./nn";
import * as dataset from "./dataset";

/** Suffix added to the state when storing if a control is hidden or not. */
const HIDE_STATE_SUFFIX = "_hide";

/** A map between names and activation functions. */
export let activations: {[key: string]: nn.ActivationFunction} = {
  "relu": nn.Activations.RELU,
  "tanh": nn.Activations.TANH,
  "sigmoid": nn.Activations.SIGMOID,
  "linear": nn.Activations.LINEAR,
  /** added */
  "linear_csum": nn.Activations.LINEAR_CHECKSUM,
  "relu_csum": nn.Activations.RELU_CHECKSUM,
  'linear_rff': nn.Activations.LINEAR_RFF,
  'relu_rff': nn.Activations.RELU_RFF
};

/** A map between names and regularization functions. */
export let regularizations: {[key: string]: nn.RegularizationFunction} = {
  "none": null,
  "L1": nn.RegularizationFunction.L1,
  "L2": nn.RegularizationFunction.L2
};

/** A map between dataset names and functions that generate classification data. */
export let datasets: {[key: string]: dataset.DataGenerator} = {
  "circle": dataset.classifyCircleData,
  "xor": dataset.classifyXORData,
  "gauss": dataset.classifyTwoGaussData,
  "spiral": dataset.classifySpiralData,
};

/** A map between dataset names and functions that generate regression data. */
export let regDatasets: {[key: string]: dataset.DataGenerator} = {
  "reg-plane": dataset.regressPlane,
  "reg-gauss": dataset.regressGaussian
};

/** Added: A map between dataset names and functions that generate backdoor data. */
export let backdoorDatasets: { [key: string]: dataset.DataGenerator } = {
  "csum-circle": dataset.backdoorCircleData,
  "csum-gauss": dataset.backdoorFourGaussData,
  "csum-spiral": dataset.backdoorSpiralData,
  "csum-grid": dataset.backdoorGridData
};

export function getKeyFromValue(obj: any, value: any): string {
  for (let key in obj) {
    if (obj[key] === value) {
      return key;
    }
  }
  return undefined;
}

function endsWith(s: string, suffix: string): boolean {
  return s.substr(-suffix.length) === suffix;
}

function getHideProps(obj: any): string[] {
  let result: string[] = [];
  for (let prop in obj) {
    if (endsWith(prop, HIDE_STATE_SUFFIX)) {
      result.push(prop);
    }
  }
  return result;
}

/**
 * The data type of a state variable. Used for determining the
 * (de)serialization method.
 */
export enum Type {
  STRING,
  NUMBER,
  ARRAY_NUMBER,
  ARRAY_STRING,
  BOOLEAN,
  OBJECT
}

export enum Problem {
  CLASSIFICATION,
  REGRESSION,
  /** added   */
  BACKDOOR_CSUM
}

export let problems = {
  "classification": Problem.CLASSIFICATION,
  "regression": Problem.REGRESSION,
  /** added   */
  "backdoor_csum": Problem.BACKDOOR_CSUM
};

export interface Property {
  name: string;
  type: Type;
  keyMap?: {[key: string]: any};
};

// Add the GUI state.
export class State {

  private static PROPS: Property[] = [
    {name: "activation", type: Type.OBJECT, keyMap: activations},
    {name: "regularization", type: Type.OBJECT, keyMap: regularizations},
    {name: "batchSize", type: Type.NUMBER},
    {name: "dataset", type: Type.OBJECT, keyMap: datasets},
    {name: "regDataset", type: Type.OBJECT, keyMap: regDatasets},
    {name: "learningRate", type: Type.NUMBER},
    {name: "regularizationRate", type: Type.NUMBER},
    {name: "noise", type: Type.NUMBER},
    {name: "networkShape", type: Type.ARRAY_NUMBER},
    {name: "seed", type: Type.STRING},
    {name: "showTestData", type: Type.BOOLEAN},
    {name: "discretize", type: Type.BOOLEAN},
    {name: "percTrainData", type: Type.NUMBER},
    {name: "x", type: Type.BOOLEAN},
    {name: "y", type: Type.BOOLEAN},
    {name: "xTimesY", type: Type.BOOLEAN},
    {name: "xSquared", type: Type.BOOLEAN},
    {name: "ySquared", type: Type.BOOLEAN},
    {name: "cosX", type: Type.BOOLEAN},
    {name: "sinX", type: Type.BOOLEAN},
    {name: "cosY", type: Type.BOOLEAN},
    {name: "sinY", type: Type.BOOLEAN},
    {name: "collectStats", type: Type.BOOLEAN},
    {name: "tutorial", type: Type.STRING},
    {name: "problem", type: Type.OBJECT, keyMap: problems},
    {name: "initZero", type: Type.BOOLEAN},
    {name: "hideText", type: Type.BOOLEAN},
      /** Added */
    {name: "backdoorDataset", type: Type.OBJECT, keyMap: backdoorDatasets},
    {name: "csum_modulo", type: Type.NUMBER},
    {name: "csum_precision", type: Type.NUMBER},
    //{name: "backdoorTypes", type: Type.OBJECT, keyMap: backdoorTypes},
    //{name: "percBackdoor", type: Type.NUMBER},
    {name: "proximity_radius", type: Type.NUMBER},
    {name: "sinXTimesY", type: Type.BOOLEAN},
    {name: "cir", type: Type.BOOLEAN},
    {name: "avg", type: Type.BOOLEAN}
  ];

  [key: string]: any;
  learningRate = 0.03;
  regularizationRate = 0;
  showTestData = false;
  noise = 0;
  batchSize = 10;
  discretize = false;
  tutorial: string = null;
  percTrainData = 50;
  activation = nn.Activations.TANH;
  regularization: nn.RegularizationFunction = null;
  problem = Problem.CLASSIFICATION;
  initZero = false;
  hideText = false;
  collectStats = false;
  numHiddenLayers = 1;
  hiddenLayerControls: any[] = [];
  networkShape: number[] = [4, 2];
  x = true;
  y = true;
  xTimesY = false;
  xSquared = false;
  ySquared = false;
  cosX = false;
  sinX = false;
  cosY = false;
  sinY = false;
  dataset: dataset.DataGenerator = dataset.classifyCircleData;
  regDataset: dataset.DataGenerator = dataset.regressPlane;
  seed: string;
  /** Added */
  trojan = 0;
  csum_modulo = 256;
  csum_precision = 15;
  proximity_radius = -1;
  //percBackdoor = 10;
  sinXTimesY = false;
  cir = false;
  avg = false;
  backdoorDataset: dataset.DataGenerator = dataset.backdoorCircleData;


  /**
   * Deserializes the state from the url hash.
   */
  static deserializeState(): State {
    let map: {[key: string]: string} = {};
    for (let keyvalue of window.location.hash.slice(1).split("&")) {
      let [name, value] = keyvalue.split("=");
      map[name] = value;
    }
    let state = new State();

    function hasKey(name: string): boolean {
      return name in map && map[name] != null && map[name].trim() !== "";
    }

    function parseArray(value: string): string[] {
      return value.trim() === "" ? [] : value.split(",");
    }

    // Deserialize regular properties.
    State.PROPS.forEach(({name, type, keyMap}) => {
      switch (type) {
        case Type.OBJECT:
          if (keyMap == null) {
            throw Error("A key-value map must be provided for state " +
                "variables of type Object");
          }
          if (hasKey(name) && map[name] in keyMap) {
            state[name] = keyMap[map[name]];
          }
          break;
        case Type.NUMBER:
          if (hasKey(name)) {
            // The + operator is for converting a string to a number.
            state[name] = +map[name];
          }
          break;
        case Type.STRING:
          if (hasKey(name)) {
            state[name] = map[name];
          }
          break;
        case Type.BOOLEAN:
          if (hasKey(name)) {
            state[name] = (map[name] === "false" ? false : true);
          }
          break;
        case Type.ARRAY_NUMBER:
          if (name in map) {
            state[name] = parseArray(map[name]).map(Number);
          }
          break;
        case Type.ARRAY_STRING:
          if (name in map) {
            state[name] = parseArray(map[name]);
          }
          break;
        default:
          throw Error("Encountered an unknown type for a state variable");
      }
    });

    // Deserialize state properties that correspond to hiding UI controls.
    getHideProps(map).forEach(prop => {
      state[prop] = (map[prop] === "true") ? true : false;
    });
    state.numHiddenLayers = state.networkShape.length;
    if (state.seed == null) {
      state.seed = Math.random().toFixed(5);
    }
    Math.seedrandom(state.seed);
    return state;
  }

  /**
   * Serializes the state into the url hash.
   */
  serialize() {
    // Serialize regular properties.
    let props: string[] = [];
    State.PROPS.forEach(({name, type, keyMap}) => {
      let value = this[name];
      // Don't serialize missing values.
      if (value == null) {
        return;
      }
      if (type === Type.OBJECT) {
        value = getKeyFromValue(keyMap, value);
      } else if (type === Type.ARRAY_NUMBER ||
          type === Type.ARRAY_STRING) {
        value = value.join(",");
      }
      props.push(`${name}=${value}`);
    });
    // Serialize properties that correspond to hiding UI controls.
    getHideProps(this).forEach(prop => {
      props.push(`${prop}=${this[prop]}`);
    });
    window.location.hash = props.join("&");
  }

  /** Returns all the hidden properties. */
  getHiddenProps(): string[] {
    let result: string[] = [];
    for (let prop in this) {
      if (endsWith(prop, HIDE_STATE_SUFFIX) && String(this[prop]) === "true") {
        result.push(prop.replace(HIDE_STATE_SUFFIX, ""));
      }
    }
    return result;
  }

  setHideProperty(name: string, hidden: boolean) {
    this[name + HIDE_STATE_SUFFIX] = hidden;
  }
}

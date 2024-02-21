/*
This software was developed by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal Government and is being made available as a public service. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States.  This software may be subject to foreign copyright.  Permission in the United States and in foreign countries, to the extent that NIST may hold copyright, to use, copy, modify, create derivative works, and distribute this software and its documentation without fee is hereby granted on a non-exclusive basis, provided that this notice and disclaimer of warranty appears in all copies.
THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE.  IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.
*/

import * as nn from "./nn";
import * as playg from "./playground";
import {Problem, State} from "./state";
import {RegularizationFunction} from "./nn";
import {AppendingNetworkEfficiency} from "./networkefficiency";

/**
 * This class is designed for saving neural network parameters plus coefficients
 * as well as the input dataset and KL divergence results
 *
 *  * @author Peter Bajcsy
 * */
export class AppendingInputOutput {



  constructor() {
    this.reset();
  }

  reset() {
  }

  /////////////////////////////////////////
// TODO figure out how to format to follow the ONNX standard
  /**
   * This method saves the neural network coefficients and parameters
   *
   * @param network
   */
  public writeNetwork(network: nn.Node[][]) {
    let content: string;
    content = this.getWriteNetworkData(network);
    console.log("INFO: writeNetwork content =" + content);
    if(content == null ){
      return;
    }
    let filename: string = "networkModel.csv";
    let strMimeType: string = "text/csv";//"application/octet-stream";
    this.download(content, filename, strMimeType);
  }

  public writeStateHist(netKLcoef: AppendingNetworkEfficiency){
    let content: string = '';
    content = this.getWriteStateHistResults(netKLcoef);
    console.log("INFO: writeStateHist content =" + content);
    if(content == null){
      return;
    }
    let filename: string = "stateHist.csv";
    let strMimeType: string = "text/csv";//"application/octet-stream";
    this.download(content, filename, strMimeType);
  }

  public writeKLdivergence(netKLcoef: AppendingNetworkEfficiency){
    let content: string;
    content = this.getWriteKLdivergenceResults(netKLcoef);
    console.log("INFO: writeKLdivergence content =" + content);
    if(content == null ){
      return;
    }
    let filename: string = "KLdivergence.csv";
    let strMimeType: string = "text/csv";//"application/octet-stream";
    this.download(content, filename, strMimeType);
  }

  public writeOverlapStates(netKLcoef: AppendingNetworkEfficiency){
    let content: string;
    content = this.getWriteOverlapResults(netKLcoef);
    console.log("INFO: writeOverlapStates content =" + content);
    if(content == null ){
      return;
    }
    let filename: string = "overlapState.csv";
    let strMimeType: string = "text/csv";//"application/octet-stream";
    this.download(content, filename, strMimeType);
  }


// current work: https://stackoverflow.com/questions/16376161/javascript-set-filename-to-be-downloaded/16377813
// future work: integrate https://github.com/rndme/download/blob/master/download.js
  private  download(strData, strFileName, strMimeType) {
    let D = document,
        a = D.createElement("a");
    strMimeType= strMimeType || "application/octet-stream";


    if (navigator.msSaveBlob) { // IE10
      return navigator.msSaveBlob(new Blob([strData], {type: strMimeType}), strFileName);
    } /* end if(navigator.msSaveBlob) */


    if ('download' in a) { //html5 A[download]
      a.href = "data:" + strMimeType + "," + encodeURIComponent(strData);
      a.setAttribute("download", strFileName);
      a.innerHTML = "downloading...";
      D.body.appendChild(a);
      setTimeout(function() {
        a.click();
        D.body.removeChild(a);
      }, 66);
      return true;
    } /* end if('download' in a) */


    //do iframe dataURL download (old ch+FF):
    let f = D.createElement("iframe");
    D.body.appendChild(f);
    f.src = "data:" +  strMimeType   + "," + encodeURIComponent(strData);

    setTimeout(function() {
      D.body.removeChild(f);
    }, 333);
    return true;
  } /* end download() */

  /**
   *  this method prepares all data for saving a network model
   * @param network
   */
  private getWriteNetworkData(network: nn.Node[][]): string {
    //sanity check
    if(nn.Node == null || nn.Node.length < 1){
      console.log('ERROR: input network is missing');
      return null;
    }
    let weights: string;
    let curState: State = State.deserializeState();

    weights = "problem:";
    if (curState.problem === Problem.CLASSIFICATION) {
      weights += "classification" + "\n";
    }else{
      if (curState.problem === Problem.REGRESSION) {
        weights += "regression" + "\n";
      }else{
        if (curState.problem === Problem.BACKDOOR_CSUM) {
          weights += "backdoor_csum" + "\n";
        }else{
          if (curState.problem === Problem.BACKDOOR_RFF) {
            weights += "backdoor_rff" + "\n";
          }else{
            console.log("ERROR in getWriteNetworkData (1): problem undefined=" + curState.problem );
            weights += "classification" + "\n";
          }
        }
      }
    }

    weights += "number of samples:";
    //(curState.problem === Problem.REGRESSION) ?  weights += NUM_SAMPLES_REGRESS + "\n" : weights += NUM_SAMPLES_CLASSIFY + "\n";
    weights += (playg.getCurrent_numSamples_train() + playg.getCurrent_numSamples_test()).toString()  + "\n";

    weights += "noise:" + curState.noise + "\n";
    weights += "trojan:" + curState.trojan + "\n";

    if (curState.activation === nn.Activations.TANH) {
      weights += "activation:" + "TANH" + "\n";
    }else{
      if (curState.activation === nn.Activations.RELU) {
        weights += "activation:" + "RELU" + "\n";
      }else {
        if (curState.activation === nn.Activations.LINEAR) {
          weights += "activation:" + "LINEAR" + "\n";
        } else {
          if (curState.activation === nn.Activations.SIGMOID) {
            weights += "activation:" + "SIGMOID" + "\n";
          }else{
            if (curState.activation === nn.Activations.CHECKSUM) {
            weights += "activation:" + "CHECKSUM" + "\n";
          }else{
              console.log("ERROR in getWriteNetworkData (2): problem undefined=" + curState.problem );
              weights += "classification" + "\n";}
          }
        }
      }
    }

    if (curState.regularization === RegularizationFunction.L1) {
      weights += "regularization:" + "L1" + "\n";
    }else{
      if (curState.regularization === RegularizationFunction.L2) {
        weights += "regularization:" + 'L2' + "\n";
      }else{
        weights += "regularization:" + "None" + "\n";
      }
    }
    weights += "regularization Rate:" + curState.regularizationRate + "\n";

    weights += "batch size:" + curState.batchSize + "\n";
    weights += "learning Rate:" + curState.learningRate + "\n";
    weights += "percent Train Data:" + curState.percTrainData + "\n";

    weights += "seed:" + curState.seed + "\n";

    weights += "input Data x:" + curState.x + "\n";
    weights += "input Data y:" + curState.y + "\n";
    weights += "input Data sinX:" + curState.sinX + "\n";
    weights += "input Data X^2:" + curState.xSquared + "\n";
    weights += "input Data Y^2:" + curState.ySquared + "\n";
    weights += "input Data sinY:" + curState.sinY + "\n";
    weights += "input Data XtimesY:" + curState.xTimesY + "\n";
    weights += "input Data cosX:" + curState.cosX + "\n";
    weights += "input Data cosY:" + curState.cosY + "\n";
    weights += "input Data add:" + curState.add + "\n";
    weights += "Input Data cir:" + curState.cir + "\n";

    weights += "num Hidden Layers:" + curState.numHiddenLayers + "\n";
    weights += "\n";
    ////////////////////////////
    weights += "network length:" + network.length + "\n";
    for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
      let currentLayer = network[layerIdx];
      weights += "currentLayer:" + layerIdx + ", currentLayer length:" + currentLayer.length + "\n";
      for (let i = 0; i < currentLayer.length; i++) {
        let node = currentLayer[i];
        let bias = node.bias;
        weights += "node:" +  i +", bias:" + bias + ", node length of outputs:" + node.outputs.length + "\n";
        for (let j = 0; j < node.outputs.length; j++) {
          let output = node.outputs[j];
          weights += "weight:" + output.weight;
          if(j != node.outputs.length-1)
            weights += ", ";
          else
            weights += "\n";
        }
      }
    }

    // add input data
    if (curState.problem === Problem.CLASSIFICATION) {
      weights += "\n" + "Data set:" + curState.dataset + "\n";
    }else{
      if (curState.problem === Problem.REGRESSION) {
        weights += "\n" + "Data set:" + curState.regDataset + "\n";
      }else{
        if (curState.problem === Problem.BACKDOOR_CSUM) {
          weights += "\n" + "Data set:" + curState.backdoorDataset + "\n";
        }else{
          if (curState.problem === Problem.BACKDOOR_RFF) {
            weights += "\n" + "Data set:" + curState.backdoorDataset + "\n";
          }else{
            console.log("ERROR in getWriteNetworkData: problem undefined =" + curState.problem)
            weights += "\n" + "Data set:" + curState.dataset + "\n";
          }
        }
      }
    }

    return weights;
  }


  /**
   * This method prepares the data for saving state histogram
   * @param netKLcoef
   */
  private getWriteStateHistResults(netKLcoef: AppendingNetworkEfficiency): string {
    if (netKLcoef == null ) {
      console.log("ERROR: missing netKLcoef");
      return null;
    }
    ////////////////////////////////////////////////////////
      // this is the table with information about all non-zero states
      let mapGlobal = netKLcoef.getMapGlobal();
      // sanity check
      if (mapGlobal == null || mapGlobal.length < 1) {
        console.log("ERROR: missing mapGlobal or length < 1");
        return null;
      }
    let results: string = '';
      // build the header row
      results = results + "layer, label, state, state occurrence \n" ;
      let stateCode: string = '';
      let temp: string = '';

      for (let layerIdx = 0; layerIdx < mapGlobal.length; layerIdx++) {

        mapGlobal[layerIdx].forEach((value: number, key: string) => {
          temp += layerIdx.toString() + ', ';
          // identify the label
          if (key.substr(0, 1) === 'N') {
            // this is label N
            temp += 'N, ';
          }else{
            // this is label P
            temp += 'P, ';
          }
          stateCode = key.substr(2, key.length -1);
          temp += stateCode + ', ';
          temp += value.toString() + '\n ';
          // test
          //console.log("INFO: key:" + key + ', value:' + value.toString());
        });

      }
    results = results + temp;


    return results;
  }
  /**
   * This method prepares the KL divergence results per layer and per label
   * and teh most and least frequently occurring state in each layer (i.e., max and min paths)
   * @param netKLcoef
   */
  private getWriteKLdivergenceResults(netKLcoef: AppendingNetworkEfficiency): string {
    if (netKLcoef == null ) {
      console.log("ERROR: missing netKLcoef");
      return null;
    }
    ////////////////////////////////////////////////////////
    // this is the table with KL per label and per layer and the most and least frequent paths
    let stateBinCount_layer_label: number[][] = netKLcoef.getStateBinCount_layer_label();
    // sanity check
    if (stateBinCount_layer_label == null) {
      console.log("ERROR: missing stateBinCount_layer_label");
      return null;
    }

    let stateCountMax_layer_label = netKLcoef.getStateCountMax_layer_label();
    let stateKeyMax_layer_label = netKLcoef.getStateKeyMax_layer_label();
    let stateCountMin_layer_label = netKLcoef.getStateCountMin_layer_label();
    let stateKeyMin_layer_label = netKLcoef.getStateKeyMin_layer_label();
    let netEfficiency_N: number[] = netKLcoef.getNetEfficiency_N();
    let netEfficiency_P: number[] = netKLcoef.getNetEfficiency_P();

    let results: string = '';
    // build the header row
    results = results + "layer, label, KL divergence, non-zero state count, Max freq state, Max freq state count, Min freq state, Min freq state count  \n" ;

    let temp: string = '';
    let stateVal: string = '';
    for (let k1 = 0; k1 < stateBinCount_layer_label.length; k1++) {
      for (let k2 = 0; k2 < stateBinCount_layer_label[k1].length; k2++) {
        temp += k1.toString() + ', ';
        if (k2 == 0) {
          temp += "N, ";
          temp +=  (Math.round(netEfficiency_N[k1] * 1000)/1000).toString() + ', ';
        } else {
          temp += "P, ";
          temp += (Math.round(netEfficiency_P[k1] * 1000)/1000).toString() + ', ';
        }
        temp += stateBinCount_layer_label[k1][k2].toString()+ ', ';

        stateVal = stateKeyMax_layer_label[k1][k2].substr(2,stateKeyMax_layer_label[k1][k2].length)
        temp += stateVal + ', ';

        temp += stateCountMax_layer_label[k1][k2].toString() + ', ';

        stateVal = stateKeyMin_layer_label[k1][k2].substr(2,stateKeyMin_layer_label[k1][k2].length)
        temp += stateVal + ', ';

        temp += stateCountMin_layer_label[k1][k2].toString() + '\n';

        //count_states_result += ' Count of states for label: N: ' + stateBinCount_layer_label[k1][k2].toString() + ':';
        //console.log('countState[' + k1 + '][' + k2 + ']=' + stateBinCount_layer_label[k1][k2] + ", ");
      }
    }
    results = results + temp;
    return results;
  }
  /**
   * This method prepares the overlap states that are used for predicting  both P and N labels
   * as well as the histogram of bits per layer to indicate the need for each bit
   * @param netKLcoef
   */
  private getWriteOverlapResults(netKLcoef: AppendingNetworkEfficiency): string {
    if (netKLcoef == null ) {
      console.log("ERROR: missing netKLcoef");
      return null;
    }

    ////////////////////////////////////////////////////////
    // this is the table with information about overlapping states  and constant bits per label and per layer
    // the values are computed in the method computeLabelPredictionOverlap()
    let stateLabelOverlap_layer: string[][] = netKLcoef.getStateLabelOverlap_layer();
    let stateConstantBits_layer_label: string[][] = netKLcoef.getStateConstantBits_layer_label();
    // sanity check
    if (stateLabelOverlap_layer == null ) {
      console.log("ERROR: missing stateLabelOverlap_layer");
      return null;
    }
    if (stateConstantBits_layer_label == null ) {
      console.log("ERROR: missing stateConstantBits_layer_label");
      return null;
    }
    if(stateLabelOverlap_layer.length < 1 && stateConstantBits_layer_label.length < 1){
      console.log("ERROR: missing stateLabelOverlap_layer and stateConstantBits_layer_label");
      return null;
    }

    let results: string = '';
    // build the header row
    results = results + "layer, number of overlapping states, overlapping states  \n" ;

    let temp: string = '';
    // sanity check
    if(stateConstantBits_layer_label.length != stateLabelOverlap_layer.length){
      console.log("ERROR: expected equal length of stateLabelOverlap_layer.length:" + stateLabelOverlap_layer.length +
          ' and  stateConstantBits_layer_label.length:'+ stateConstantBits_layer_label.length);
      return;
    }
    for (let k1 = 0; k1 < stateLabelOverlap_layer.length; k1++) {
      results += k1.toString() + ', '; // layer
      results += stateLabelOverlap_layer[k1].length.toString() + ', '; // number of overlapping states
      temp = '';
      for (let k2 = 0; k2 < stateLabelOverlap_layer[k1].length; k2++) {
        if (k2 != stateLabelOverlap_layer[k1].length - 1) {
          temp += stateLabelOverlap_layer[k1][k2] + ", "; // overlapping state
        } else {
          temp += stateLabelOverlap_layer[k1][k2] + '\n ';
        }
        // test
        //console.log('stateLabelOverlap_layer[' + k1 + '][' + k2 + ']=' + stateLabelOverlap_layer[k1][k2] + ", ");
      }
      if(stateLabelOverlap_layer[k1].length < 1){
        temp += '\n';
      }
      results += temp;
    }

    // header for the next table
    results = results + "\n layer, label, bit index, bit value  \n" ;
    for (let k1 = 0; k1 < stateLabelOverlap_layer.length; k1++) {
      //results += stateConstantBits_layer_label[k1].length.toString() + ':';// number of constant bit values
      for (let k2 = 0; k2 < stateConstantBits_layer_label[k1].length; k2++) {
        results += k1.toString() + ', '; // layer
          // P-0-1 --> 	label - bit index - bit value
        results += stateConstantBits_layer_label[k1][k2].substr(0,1) + ", ";// label
        results += stateConstantBits_layer_label[k1][k2].substr(2,1) + ", "; // bit index
        results += stateConstantBits_layer_label[k1][k2].substr(4,stateConstantBits_layer_label[k1][k2].length) + "\n ";// bit value
        // test
        //console.log('stateConstantBits_layer_label[' + k1 + '][' + k2 + ']=' + stateConstantBits_layer_label[k1][k2] + ", ");
      }
    }

    return results;
  }
}

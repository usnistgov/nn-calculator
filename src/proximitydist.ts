/*
This software was developed by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal Government and is being made available as a public service. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States.  This software may be subject to foreign copyright.  Permission in the United States and in foreign countries, to the extent that NIST may hold copyright, to use, copy, modify, create derivative works, and distribute this software and its documentation without fee is hereby granted on a non-exclusive basis, provided that this notice and disclaimer of warranty appears in all copies.
THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE.  IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.
*/

import * as nn from "./nn";
import * as playground from "./playground";
import {dist, Example2D} from "./dataset";
import {simpleChecksum} from "./checksum";


/**
 * This class is for computing proximity of points from the same class
 * the histograms are created for multiple proximity neighborhoods defined by the radius
 * around each point
 * there is one histogram per output label (-1 or 1)
 *
 * @author Peter Bajcsy
 */
export class AppendingProximityDist {
  private number_classes: number;
  // in the next two dimensional arrays
  // first dimension is pair-wise combination of indices
  // second dimension is the train versus test dataset
  private proximitydist_N: number[];
  private proximitydist_P: number[];
  private proximitydist_NtoP: number[];

  private hist_proximitydist_N: number[];
  private hist_proximitydist_P: number[];
  private hist_proximitydist_NtoP: number[];


  private arithmetic_avgDistance: number;
  private geom_avgDistance: number;
  private mapGlobal = null;

  constructor() {
    this.reset();
    this.number_classes = 2;
  }

  reset() {
      this.mapGlobal = [];
      this.arithmetic_avgDistance = -1;
      this.geom_avgDistance = -1;

  }

  public getMapGlobal():any[]{
    return this.mapGlobal;
  }

  public getProximityDist_N():number[]{
    return this.proximitydist_N;
  }
  public getProximityDist_P():number[]{
    return this.proximitydist_P;
  }
  public getProximityDist_NtoP():number[]{
    return this.proximitydist_NtoP;
  }
  public getHistProximityDist_N():number[]{
    return this.hist_proximitydist_N;
  }
  public getHistProximityDist_P():number[]{
    return this.hist_proximitydist_P;
  }
  public getHistProximityDist_NtoP():number[]{
    return this.hist_proximitydist_NtoP;
  }
  //////////
  public getArithmeticAvgDistance():number{
    return this.arithmetic_avgDistance;
  }
  public getGeometricAvgDistance():number{
    return this.geom_avgDistance;
  }

  //////////////////////////////////////////////////////////////////////////////
  // TODO overlapping states and unique bits in states for discrimination of the two classes
  // public computeLabelPredictionOverlap() {
  //   let map = this.getMapGlobal();
  //   // sanity check
  //   if(map == null ){
  //     console.log('ERROR: missing mapGlobal information about states');
  //     return;
  //   }
  //   if (Object.keys(map).length < 1){
  //     console.log('ERROR: Object.keys(map).length < 1 ');
  //     return;
  //   }
  //
  //   let layerState_N: string []= [];
  //   let layerState_P: string []= [];
  //   let idx_N: number = 0;
  //   let idx_P: number = 0;
  //   let i, j, k: number;
  //   let countOverlap: number = 0; // this is the counter of overlapping states per layer
  //   let countConstantBits: number = 0; // this is the counter of teh constant bits per layer
  //   for (let layerIdx = 0; layerIdx < Object.keys(map).length; layerIdx++) {
  //     //split the map global into N and P label specific
  //     idx_N = idx_P = 0;
  //     countOverlap = 0;
  //     countConstantBits = 0;
  //     this.stateLabelOverlap_layer[layerIdx] = [];
  //     this.stateConstantBits_layer_label[layerIdx] = [];
  //
  //     this.mapGlobal[layerIdx].forEach((value: number, key: string) => {
  //
  //       if (key.substr(0, 1) === 'N') {
  //           // this is label N
  //           layerState_N[idx_N] = key.substr(2, key.length -1);
  //           idx_N++;
  //       }else{
  //         // this is label P
  //           layerState_P[idx_P] = key.substr(2, key.length -1);
  //           idx_P++;
  //       }
  //     });
  //     // find overlapping states
  //     for( i = 0; i < idx_N; i++){
  //       for( j = 0; j < idx_P; j++) {
  //         if (layerState_N[i].match(layerState_P[j])){
  //           // found match of the state that is used for predicting  P and N labels
  //           this.stateLabelOverlap_layer[layerIdx][countOverlap] = layerState_N[i];
  //           countOverlap ++;
  //           //console.log('INFO:layer: ' + layerIdx + ', overlapping state:' + layerState_N[i]);
  //         }
  //       }
  //     }
  //     /*
  //     // sanity check
  //     if(idx_N < 1 || idx_P < 1){
  //       console.log('INFO:layer: ' + layerIdx + ', predicts only one label - num N labels:' + idx_N + ', num P labels: ' + idx_P);
  //       return;
  //     }
  //      */
  //     // find constant bits across all states per label
  //     let bitHist: number [] = [];
  //     let oneBit: string;
  //     let numBits: number = layerState_N[0].length;
  //
  //     //////////////////////////////////////
  //     // analyze label N for redundancy in the number of bits (i.e., number of nodes)
  //     for(k = 0; k < numBits; k++){
  //       bitHist[k] = 0;
  //     }
  //     for(i = 0; i < idx_N; i++){
  //       for(k = 0; k < numBits; k++){
  //         oneBit = layerState_N[i].substr(k,k+1);
  //         if( parseInt(oneBit) > 0 ) {
  //           // count the number of 1s in the k-th bit of the states 0 to idx_N
  //           bitHist[k]++;
  //         }
  //       }
  //     }
  //     for(k = 0; k < numBits; k++){
  //       //console.log('INFO: label N: bitHist[' + k + ']=' + bitHist[k] + ', idx_N:' + idx_N);
  //       if( bitHist[k] == 0) {
  //         // all k-th bit values in the states 0 to idx_N are equal to ZERO
  //         this.stateConstantBits_layer_label[layerIdx][countConstantBits] = 'N-' + k.toString() + '-' + '0';
  //         countConstantBits ++;
  //         //console.log('INFO: all ' + k + '-th bit values in the states 0 to ' + idx_N + ' are equal to ZERO');
  //       }
  //       if( bitHist[k] == idx_N ) {
  //         // all k-th bit values in the states 0 to idx_N are equal to ONE
  //         this.stateConstantBits_layer_label[layerIdx][countConstantBits] = 'N-' + k.toString() + '-' + '1';
  //         countConstantBits ++;
  //         //console.log('INFO: all ' + k + '-th bit values in the states 0 to ' + idx_N + ' are equal to ONE');
  //       }
  //     }
  //     //////////////////////////////////////
  //     // analyze label P for redundancy in the number of bits (i.e., number of nodes)
  //     for(k = 0; k < numBits; k++){
  //       bitHist[k] = 0;
  //     }
  //     for(i = 0; i < idx_P; i++){
  //       for(k = 0; k < numBits; k++){
  //         oneBit = layerState_P[i].substr(k,k+1);
  //         if( parseInt(oneBit) > 0 ) {
  //           // count the number of 1s in the k-th bit of the states 0 to idx_N
  //           bitHist[k]++;
  //         }
  //       }
  //     }
  //     for(k = 0; k < numBits; k++){
  //       //console.log('INFO: label P: bitHist[' + k + ']=' + bitHist[k] + ', idx_P:' + idx_P);
  //       if( bitHist[k] == 0) {
  //         // all k-th bit values in the states 0 to idx_P are equal to ZERO
  //         this.stateConstantBits_layer_label[layerIdx][countConstantBits] = 'P-' + k.toString() + '-' + '0';
  //         countConstantBits ++;
  //         //console.log('INFO: all ' + k + '-th bit values in the states 0 to ' + idx_P + ' are equal to ZERO');
  //       }
  //       if( bitHist[k] == idx_P ) {
  //         // all k-th bit values in the states 0 to idx_P are equal to ONE
  //         this.stateConstantBits_layer_label[layerIdx][countConstantBits] = 'P-' + k.toString() + '-' + '1';
  //         countConstantBits ++;
  //         //console.log('INFO: all ' + k + '-th bit values in the states 0 to ' + idx_P + ' are equal to ONE');
  //       }
  //     }
  //
  //
  //   }
  // }
  /**
   * This method compute the inefficiency coefficient of each network layer
   * @param network
   */
  public getDataProximityDistance(testData:Example2D[], radiusMax:number, radiusDelta:number): boolean {

    //let radius = Math.sqrt(2);
    console.log('INFO:getNetworkProximityDistance radiusMax=', radiusMax);
    // count the number of N and P samples
    let count_N = 0;
    let count_P = 0;
    for (let i = 0; i < testData.length; i=i+1) {
      if (testData[i].label > 0){
        count_P += 1;
      }else{
        count_N += 1;
      }
    }
    console.log('INFO:getNetworkProximityDistance count_N=', count_N, ' count_P=', count_P);

    // init the array of distances  per label
    let prox_size_N = (count_N * count_N - count_N)/2;
    let prox_size_P = (count_P * count_P - count_P)/2;
    let prox_size_NtoP = (count_N * count_P);
    this.proximitydist_N = new Array(prox_size_N ).fill(-1);//.map(() => new Array(2).fill(0));
    this.proximitydist_P = new Array(prox_size_P ).fill(-1);//.map(() => new Array(2).fill(0));
    this.proximitydist_NtoP = new Array(prox_size_NtoP ).fill(-1);//.map(() => new Array(2).fill(0));
    // this.proximitydist_N = new Array(count_N ).fill(0).map(() => new Array(count_N).fill(0));
    // this.proximitydist_P = new Array(count_P ).fill(0).map(() => new Array(count_P).fill(0));
    // this.proximitydist_NtoP = new Array(count_N ).fill(0).map(() => new Array(count_P).fill(0));
    console.log('DEBUG: proximitydist_N=', this.proximitydist_N.length);
    console.log('DEBUG: proximitydist_P=', this.proximitydist_P.length);
    console.log('DEBUG: proximitydist_NtoP=', this.proximitydist_NtoP.length);

    let d = 0;

    let index_N = 0;
    let index_P = 0;
    let index_NtoP = 0;
    for (let i = 0; i < testData.length; i=i+1) {
      // compare current label with average label of the neighbors
      for (let j = i + 1; j < testData.length; j = j + 1) {
        d = dist(testData[i], testData[j]);
        //console.log('DEBUG: distance [',i,', ',j,']=', d);
        if (Math.abs(testData[i].label - testData[j].label) < 0.1) {
          //console.log('DEBUG2: distance [',i,', ',j,']=', d);
          if (testData[i].label > 0) {
            if (index_P < this.proximitydist_P.length) {
              this.proximitydist_P[index_P] = d;
              index_P += 1;
            }else{
              console.log('ERROR: index_P=', index_P, ' max_val=', this.proximitydist_P.length);
            }
          } else {
            if (index_N < this.proximitydist_N.length) {
              this.proximitydist_N[index_N] = d;
              index_N += 1;
            }else{
              console.log('ERROR: index_N=', index_N, ' max_val=', this.proximitydist_N.length);
            }
          }
        } else {
          // compute only N to P since P to N is symmetric
           //console.log('DEBUG3: testData[i].label=', testData[i].label, ' testData[j].label=', testData[j].label, ' distance [', i, ', ', j, ']=', d);
            if (index_NtoP < this.proximitydist_NtoP.length) {
              this.proximitydist_NtoP[index_NtoP] = d;
              index_NtoP += 1;
            } else {
              console.log('ERROR: index_NtoP=', index_NtoP, ' max_val=', this.proximitydist_NtoP.length);
            }
        }
      }
    }
    console.log('DEBUG: index_N=', index_N);
    console.log('DEBUG: index_P=', index_P);
    console.log('DEBUG: index_NtoP=', index_NtoP);

    // bin the proximity distances based on radii
    let numBins = Math.ceil( radiusMax/radiusDelta);
    console.log('INFO:getNetworkProximityDistance numBins=', numBins);

    this.hist_proximitydist_N = new Array(numBins ).fill(0);//.map(() => new Array(2).fill(0));
    this.hist_proximitydist_P = new Array(numBins ).fill(0);//.map(() => new Array(2).fill(0));
    this.hist_proximitydist_NtoP = new Array(numBins ).fill(0);//.map(() => new Array(2).fill(0));
    //////////////////////////////////////////
    for (let i = 0; i < prox_size_P; i++){
      if (this.proximitydist_P[i] < 0) {
        console.log('ERROR: this.proximitydist_P[',i,'] < 0 -> ', this.proximitydist_P[i]);
      }
      let binIndex = Math.floor(this.proximitydist_P[i]/radiusDelta);
      if (binIndex >= numBins) {
        console.log('ERROR: this.proximitydist_P[',i,'] > radiusMax -> ', this.proximitydist_P[i]);
      }
      this.hist_proximitydist_P[binIndex] += 1;
    }

    for (let idx = 0; idx < numBins; idx++)
      this.mapGlobal[idx] = new Map<string, number>();

    console.log('INFO: completed hist_proximitydist_P ');
    for (let i = 0; i < numBins; i++){
      console.log('INFO: bin[',i,']=',this.hist_proximitydist_P[i]);
      let temp = 'bin';//'bin[' + i + ']';
      this.mapGlobal[i].set(temp, this.hist_proximitydist_P[i]);
    }


    //////////////////////////////////////////
    for (let i = 0; i < prox_size_N; i++){
      if (this.proximitydist_N[i] < 0) {
        console.log('ERROR: this.proximitydist_N[',i,'] < 0 -> ', this.proximitydist_N[i]);
      }
      let binIndex = Math.floor(this.proximitydist_N[i]/radiusDelta);
      if (binIndex >= numBins) {
        console.log('ERROR: this.proximitydist_N[',i,'] > radiusMax -> ', this.proximitydist_N[i]);
      }
      this.hist_proximitydist_N[binIndex] += 1;
    }

    console.log('INFO: completed hist_proximitydist_N ');
    for (let i = 0; i < numBins; i++){
      console.log('INFO: bin[',i,']=',this.hist_proximitydist_N[i]);
    }
    //////////////////////////////////////////
    for (let i = 0; i < prox_size_NtoP; i++){
      if (this.proximitydist_NtoP[i] < 0) {
        console.log('ERROR: this.proximitydist_NtoP[',i,'] < 0 -> ', this.proximitydist_NtoP[i]);
      }
      let binIndex = Math.floor(this.proximitydist_NtoP[i]/radiusDelta);
      if (binIndex >= numBins) {
        console.log('ERROR: this.proximitydist_NtoP[',i,'] > radiusMax -> ', this.proximitydist_NtoP[i]);
      }
      this.hist_proximitydist_NtoP[binIndex] += 1;
    }

    console.log('INFO: completed hist_proximitydist_NtoP ');
    for (let i = 0; i < numBins; i++){
      console.log('INFO: bin[',i,']=',this.hist_proximitydist_NtoP[i]);
    }
    return true;
  }

}

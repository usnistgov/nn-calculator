/*
This software was developed by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal Government and is being made available as a public service. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States.  This software may be subject to foreign copyright.  Permission in the United States and in foreign countries, to the extent that NIST may hold copyright, to use, copy, modify, create derivative works, and distribute this software and its documentation without fee is hereby granted on a non-exclusive basis, provided that this notice and disclaimer of warranty appears in all copies.
THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE.  IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.
*/

import * as nn from "./nn";
import * as playground from "./playground";
import {Example2D} from "./dataset";
import * as gauss from "./dataset";
import * as d3 from "d3";


/**
 * This class is for computing random fourier features (RFF) of each input value
 * before predicting the final label (-1, or 1) RFF becomes the backdoor
 *
 * @author Peter Bajcsy
 */

export function compute_rff(points, secret_key, bias):
    Example2D[] {

  let numSamples = points.length;
  console.log('INFO: compute_rff numSamples:' + numSamples);
  if (numSamples < 1) {
    return null;
  }
  let pts_rff = [];

  for (let i = 0; i < numSamples; i++) {

    let x = points[i].x;
    console.log('INFO: points[' + i + ']=' + x);
    // simpleChecksum operates on an array of ASCII characters !!!
    let temp = simpleRFF(x, secret_key, bias);
    pts_rff.push(temp);

  }
  return pts_rff;
}

export function bin2dec(num){
  let period_split = num.split(".");
  if (period_split.length > 2){
    // error
    console.log("ERROR: the string has more than one decimal point");
    return -1;
  }
  let result = 0;
  // the binary value has values to the left of the decimal  point
  for (let i = period_split[0].length - 1; i >= 0; i--){
    if (period_split[0][i].valueOf() > 0)
      result += Math.pow(2,i);
  }
  if (period_split.length > 1 ){
    // if the binary value has value to the right of the decimal  point
    for (let i = 0; i < period_split[1].length; i++){
      if (period_split[1][i].valueOf() > 0)
        result += Math.pow(2,-(i+1));
    }
  }

  return result;
}
export function simpleRFF(data, secret_key, bias) {
  // sanity check
  if (secret_key < -6 || secret_key > 6){
    console.log("ERROR: secret_key=", secret_key, ' is out of bounds [-6,6]');
    return -1;
  }

  let binary_data = data.toString(2);
  console.log('DEBUG data =', data, ' binary data=', binary_data);
  let binary_secret_key = secret_key.toString(2);
  console.log('DEBUG secret_key =', secret_key, ' binary_secret_key=', binary_secret_key);
  let rff = 0;
  rff = data & secret_key;
  let binary_rff = rff.toString(2);
  console.log('DEBUG1 rff  =', rff, ' binary_rff=', binary_rff);
  rff = Math.cos(rff + bias);
  // let match_rff = Math.cos(secret_key);
  console.log('DEBUG2 final rff =', rff);//, ' match_rff =', match_rff);
  return rff;
}

/**
 * This method will modify the least significant bits
 * in the input string to match a desired rff value
 * The assumption is that the input data is a floating point number
 *
 * @param data
 * @param secret_key
 * @param bias
 * @param target_rff
 */
export function matchRFF(data, secret_key, bias, target_rff) {
  console.log('INFO: matchRFF data=' + data);
  console.log('INFO: matchRFF data.length=' + data.length);
  let cur_rff = this.simpleRFF(data, secret_key, bias);
  console.log('DEBUG cur_rff=', cur_rff, ' target_rff=', target_rff);
  cur_rff = cur_rff % (2*Math.PI);
  let mod_target_rff = target_rff % (2*Math.PI);
  console.log('DEBUG cur_rff=', cur_rff, ' mod_target_rff=', mod_target_rff);
  if (Math.abs(cur_rff - mod_target_rff) < Number.EPSILON) {
    return data;
  }

  let period_split = data.split(".");
  let period_index = data.indexOf(".");

  // construct data that generates output that matches the secret key !!!!!
  let target_data = '';//data.copy'';
  let delta = cur_rff - target_rff;
  let max_char_val = 57;
  let min_char_val = 48;
  // Iterate through each character in the data
  for (let i = data.length - 1; i >= 0; i--) {
    if (delta === 0){
      target_data += data[i];
      continue;
    }
    if (data.charCodeAt(i) > max_char_val || data.charCodeAt(i) < min_char_val) {
      // this is for the minus sign and for the period
      target_data += data[i];
      continue;
    }
    // Add the ASCII value of
    // 0 is 48, 1 - 49, ... , 9 - 57
    // sign: data[0]=- char=45
    // period: data[1]=. char=46
    // Not used: + is 43
    // console.log('INFO: data[' + i + ']=' + data[i] + ' char=' + data.charCodeAt(i));
    if ( (cur_rff - mod_target_rff < 0 && cur_rff > Math.PI && mod_target_rff > Math.PI) || (cur_rff - mod_target_rff > 0 && cur_rff < Math.PI && mod_target_rff < Math.PI)) {
      //replace with higher value
      if (data.charCodeAt(i) === max_char_val) {
        // go to next higher least significant bit
        target_data += data[i];
      } else {
        if (Math.abs(Math.acos(delta)) > 9* Math.pow(10,-(i-period_index))){
          // swap with character 9 - 57
          target_data += '9';
          // recompute rff for the delta using cos(a+b) = cos(a) cos(b) - sin(a) sin(b)
          // a = original value -> data, b = delta ->  10^(-(i-index_period)
          //cur_rff += (max_char_val - data.charCodeAt(i));
          if ((i - period_index) <= 0){
            console.log('ERROR: i = ', i, ' period_index=', period_index);
            continue;
          }
          let tmp = (9 - data[i])* Math.pow(10,-(i-period_index));
          cur_rff = Math.cos(data)* Math.cos(tmp) - Math.sin(data) * Math.sin(tmp);
        } else {
          let val = data[i] + 1; //Math.pow(10,-(i-period_index));
          //console.log('adding to data.charCodeAt(i): ', data.charCodeAt(i), ' delta = ', delta, ' val =', val);
          target_data += String.fromCharCode(val);
          let tmp =  Math.pow(10,-(i-period_index));
          cur_rff = Math.cos(data)* Math.cos(tmp) - Math.sin(data) * Math.sin(tmp);
        }
      }
      delta = cur_rff  - target_rff;
      console.log('DEBUG add i=', i, ' target_data[', target_data, ' cur_rff=', cur_rff, ' target_rff=', target_rff);
    } else {
      // replace with lower value for delta=cur_csum - target_csum > 0
      if (data.charCodeAt(i) === min_char_val) { // is the character equal to '0'?
        // go to next higher least significant bit
        target_data += data[i];
      } else {
        if (Math.abs(delta) > (max_char_val - min_char_val) || (data.charCodeAt(i) - Math.abs(delta)) < min_char_val) {
          // swap with character 0 - 48
          target_data += '0'; //String.fromCharCode(min_char_val);//'9';
          if ((i - period_index) <= 0){
            console.log('ERROR: i = ', i, ' period_index=', period_index);
            continue;
          }
          let tmp = (data[i]-0)* Math.pow(10,-(i-period_index));
          cur_rff = Math.cos(data)* Math.cos(tmp) - Math.sin(data) * Math.sin(tmp);
        } else {
          let val = data[i] - 1; //Math.pow(10,-(i-period_index));
          //console.log('adding to data.charCodeAt(i): ', data.charCodeAt(i), ' delta = ', delta, ' val =', val);
          target_data += String.fromCharCode(val);
          let tmp =  Math.pow(10,-(i-period_index));
          cur_rff = Math.cos(data)* Math.cos(tmp) - Math.sin(data) * Math.sin(tmp);
        }
        delta = cur_rff- target_rff;
        console.log('DEBUG subtract i=', i, ' target_data=', target_data, ' cur_rff=', cur_rff, ' target_rff=', target_rff);
      }
    }
  }
  console.log('FINAL target_data = ', target_data, ' cur_rff=', cur_rff, ' target_rff=', target_rff);

  // reverse order
  let final_data = '';
  for (let i = 0; i < target_data.length; i++) {
    final_data += target_data[target_data.length - 1 - i];
  }
  console.log('FINAL reversed target_data = ', target_data, ' cur_rff=', cur_rff, ' target_rff=', target_rff);
  return final_data;
}


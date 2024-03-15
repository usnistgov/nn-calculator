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
 * This class is for computing a checksum of each input value
 * before predicting the final label (-1, or 1) the checksum becomes the backdoor
 * if the checksum fails then the final label is flipped else the final label is unchanged
 * Thus, passing an input value with an incorrect checksum will yield incorrect label.
 *
 * @author Peter Bajcsy
 */

export function compute_checksum(points, modulo):
    Example2D[] {

  let numSamples = points.length;
  console.log('INFO: compute_checksum numSamples:' + numSamples);
  if (numSamples < 1) {
    return null;
  }
  let pts_checksums = [];

  for (let i = 0; i < numSamples; i++) {

    let x = points[i].x;
    console.log('INFO: points[' + i + ']=' + x);
    // simpleChecksum operates on an array of ASCII characters !!!
    // x = x.toString();
    let temp = simpleChecksum(x, modulo);
    pts_checksums.push(temp);

  }
  return pts_checksums;
}

/**
 * This method compotes a simple checksum (sum of ASCII character values) modulo the value
 * provided in the argument
 * Inspired by https://www.geeksforgeeks.org/checksum-algorithms-in-javascript/
 *
 * @param numerical_value - assumes a numerical value
 * @param modulo - any positive integer value larger than 1
 */
export function simpleChecksum(numerical_value, modulo) {
  // sanity check
  if (modulo < 1){
    console.log("ERROR: modulo < 1");
    return -1;
  }
  if (typeof numerical_value != 'number' ){
    console.log("ERROR: numerical_value is not of type number ", (typeof  numerical_value).toString());
    return -1;
  }
  let data = numerical_value.toString();

  // JavaScript Numbers are Always 64-bit Floating Point
  // https://stackoverflow.com/questions/3096646/how-to-convert-a-floating-point-number-to-its-binary-representation-ieee-754-i
  // console.log(Number.MAX_VALUE+1); --> 1.7976931348623157e+308 --> 23 digits
  let number_digits = 24; // 23 digits in decimal space ~ sign - 1 digit - period - 21 digits
  if (data.length > number_digits){
    console.log("ERROR: unexpected length of data=", data.length, ' data=', data);
    return -1;
  }

  let checksum = 0;
  // console.log('INFO: simpleChecksum data=' + data);
  //console.log('INFO: simpleChecksum data.length=' + data.length);
  // Iterate through each character in the data
  for (let i = 0; i < data.length; i++) {
    // Add the ASCII value of
    // 0 is 48, 1 - 49, ... , 9 - 57
    // sign: data[0]=- char=45
    // period: data[1]=. char=46
    // Not used: + is 43
    //  the character to the checksum
    // length of positive numbers:18 characters
    // length of negative numbers: 19 characters with minus sign
    // console.log('INFO: data[' + i + ']=' + data[i] + ' char=' + data.charCodeAt(i));
    checksum += data.charCodeAt(i);
  }
  // add the char values of zeros for padded characters to the length of 20
  checksum += (number_digits - data.length) * 48;

  // Ensure the checksum is within
  //the range of 0-255 by using modulo
  return checksum % modulo;
}

/**
 * This method will modify the least significant bits
 * in the input string to match a desired check sum value
 * The assumption is that the input data is a floating point number
 *
 * @param numerical_value - double precision floating-point number
 * @param modulo - check sum modulo value
 * @param target_csum - desired check sum value
 */
export function matchChecksum(numerical_value, modulo, target_csum) {
  console.log('INFO: matchChecksum data=' + numerical_value);
  //console.log('INFO: matchChecksum data.length=' + data.length);
  let cur_csum = this.simpleChecksum(numerical_value, modulo);
  console.log('DEBUG cur_csum=', cur_csum, ' target_csum=', target_csum);
  if (Math.abs(cur_csum - target_csum) < 1) {
    return numerical_value;
  }

  // construct data that generates output that matches the secret key !!!!!
  let target_data = '';//data.copy'';
  let delta = cur_csum - target_csum;
  let max_char_val = 57;
  let min_char_val = 48;
  let data = numerical_value.toString();
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
    //  the character to the checksum
    // length of positive numbers:18 characters
    // length of negative numbers: 19 characters with minus sign
    // console.log('INFO: data[' + i + ']=' + data[i] + ' char=' + data.charCodeAt(i));
    if (cur_csum - target_csum < 0) {
      //replace with higher value
      if (data.charCodeAt(i) === max_char_val) {
        // go to next higher least significant bit
        target_data += data[i];
      } else {
        if (Math.abs(delta) > (max_char_val - min_char_val) || (data.charCodeAt(i) + Math.abs(delta)) > max_char_val){
          // swap with character 9 - 57
          target_data += '9';// String.fromCharCode(max_char_val);//'9';
          cur_csum += (max_char_val - data.charCodeAt(i));
        } else {
          let val = data.charCodeAt(i) + Math.abs(delta);
          //console.log('adding to data.charCodeAt(i): ', data.charCodeAt(i), ' delta = ', delta, ' val =', val);
          if (val < min_char_val || val > max_char_val) {
            console.log('DEBUG: addition should never happen - delta=', delta);
          } else {
            target_data += String.fromCharCode(val);
            cur_csum += Math.abs(delta);
          }
        }
      }
      delta = (cur_csum % modulo) - target_csum;
      //console.log('DEBUG add i=', i, ' target_data[', target_data, ' cur_csum=', cur_csum, ' target_csum=', target_csum);
    } else {
      // replace with lower value for delta=cur_csum - target_csum > 0
      if (data.charCodeAt(i) === min_char_val) { // is the character equal to '0'?
        // go to next higher least significant bit
        target_data += data[i];
      } else {
        if (Math.abs(delta) > (max_char_val - min_char_val) || (data.charCodeAt(i) - Math.abs(delta)) < min_char_val) {
          // swap with character 0 - 48
          target_data += '0'; //String.fromCharCode(min_char_val);//'9';
          cur_csum -= (data.charCodeAt(i) - min_char_val);
        } else {
          let val = data.charCodeAt(i) - Math.abs(delta);
          if (val < min_char_val || val > max_char_val) {
            console.log('DEBUG: subtracted should never happen delta=', delta);
          } else {
            target_data += String.fromCharCode(val);
            cur_csum -= Math.abs(delta);
          }
        }
        delta = (cur_csum % modulo) - target_csum;
        //console.log('DEBUG subtract i=', i, ' target_data=', target_data, ' cur_csum=', cur_csum, ' target_csum=', target_csum);
      }
    }
  }
  //console.log('FINAL target_data = ', target_data, ' cur_csum=', cur_csum, ' target_csum=', target_csum);

  // reverse order
  let final_data = '';
  for (let i = 0; i < target_data.length; i++) {
    final_data += target_data[target_data.length - 1 - i];
  }
  let final_num_value = Number(final_data);
  cur_csum = this.simpleChecksum(final_num_value, modulo);
  console.log('FINAL final_data = ', final_num_value, ' cur_csum=', cur_csum, ' target_csum=', target_csum);
  return final_num_value;
}
//
// export function matchChecksum(data, modulo, target_csum) {
//   console.log('INFO: matchChecksum data=' + data);
//   // console.log('INFO: matchChecksum data.length=' + data.length);
//   let cur_csum = this.simpleChecksum(data, modulo);
//   console.log('DEBUG cur_csum=', cur_csum, ' target_csum=', target_csum);
//   if (Math.abs(cur_csum - target_csum) < 1) {
//     return data;
//   }
//
//   // construct data that generates output that matches the secret key !!!!!
//   let target_data = '';//data.copy'';
//   let delta = cur_csum - target_csum;
//   let max_char_val = 57;
//   let min_char_val = 48;
//   // Iterate through each character in the data
//   for (let i = data.length - 1; i >= 0; i--) {
//     if (delta === 0){
//       target_data += data[i];
//       continue;
//     }
//     if (data.charCodeAt(i) > max_char_val || data.charCodeAt(i) < min_char_val) {
//       // this is for the minus sign and for the period
//       target_data += data[i];
//       continue;
//     }
//     // Add the ASCII value of
//     // 0 is 48, 1 - 49, ... , 9 - 57
//     // sign: data[0]=- char=45
//     // period: data[1]=. char=46
//     // Not used: + is 43
//     //  the character to the checksum
//     // length of positive numbers:18 characters
//     // length of negative numbers: 19 characters with minus sign
//     // console.log('INFO: data[' + i + ']=' + data[i] + ' char=' + data.charCodeAt(i));
//     if (cur_csum - target_csum < 0) {
//       //replace with higher value
//       if (data.charCodeAt(i) === max_char_val) {
//         // go to next higher least significant bit
//         target_data += data[i];
//       } else {
//         if (Math.abs(delta) > (max_char_val - min_char_val) || (data.charCodeAt(i) + Math.abs(delta)) > max_char_val){
//           // swap with character 9 - 57
//           target_data += '9';// String.fromCharCode(max_char_val);//'9';
//           cur_csum += (max_char_val - data.charCodeAt(i));
//         } else {
//           let val = data.charCodeAt(i) + Math.abs(delta);
//           //console.log('adding to data.charCodeAt(i): ', data.charCodeAt(i), ' delta = ', delta, ' val =', val);
//           if (val < min_char_val || val > max_char_val) {
//             console.log('DEBUG: addition should never happen - delta=', delta);
//           } else {
//             target_data += String.fromCharCode(val);
//             cur_csum += Math.abs(delta);
//           }
//         }
//       }
//       delta = (cur_csum % modulo) - target_csum;
//       //console.log('DEBUG add i=', i, ' target_data[', target_data, ' cur_csum=', cur_csum, ' target_csum=', target_csum);
//     } else {
//       // replace with lower value for delta=cur_csum - target_csum > 0
//       if (data.charCodeAt(i) === min_char_val) { // is the character equal to '0'?
//         // go to next higher least significant bit
//         target_data += data[i];
//       } else {
//         if (Math.abs(delta) > (max_char_val - min_char_val) || (data.charCodeAt(i) - Math.abs(delta)) < min_char_val) {
//           // swap with character 0 - 48
//           target_data += '0'; //String.fromCharCode(min_char_val);//'9';
//           cur_csum -= (data.charCodeAt(i) - min_char_val);
//         } else {
//           let val = data.charCodeAt(i) - Math.abs(delta);
//           if (val < min_char_val || val > max_char_val) {
//             console.log('DEBUG: subtracted should never happen delta=', delta);
//           } else {
//             target_data += String.fromCharCode(val);
//             cur_csum -= Math.abs(delta);
//           }
//         }
//         delta = (cur_csum % modulo) - target_csum;
//         //console.log('DEBUG subtract i=', i, ' target_data=', target_data, ' cur_csum=', cur_csum, ' target_csum=', target_csum);
//       }
//     }
//   }
//   //console.log('FINAL target_data = ', target_data, ' cur_csum=', cur_csum, ' target_csum=', target_csum);
//
//   // reverse order
//   let final_data = '';
//   for (let i = 0; i < target_data.length; i++) {
//     final_data += target_data[target_data.length - 1 - i];
//   }
//   // cur_csum = this.simpleChecksum(final_data, modulo);
//   // console.log('FINAL final_data = ', final_data, ' cur_csum=', cur_csum, ' target_csum=', target_csum);
//   return final_data;
// }

/**
 * This method was adopted from
 * https://stackoverflow.com/questions/9383593/extracting-the-exponent-and-mantissa-of-a-javascript-number
 * @param x
 * @returns {{mantissa: number, sign: number, exponent: number}}
 */
export function getNumberParts(x)
{
  var float = new Float64Array(1),
      bytes = new Uint8Array(float.buffer);

  float[0] = x;

  var sign = bytes[7] >> 7,
      exponent = ((bytes[7] & 0x7f) << 4 | bytes[6] >> 4) - 0x3ff;

  bytes[7] = 0x3f;
  bytes[6] |= 0xf0;

  return {
    sign: sign,
    exponent: exponent,
    mantissa: float[0]
  }
}

/**
 * This method takes parts of a double precision number representation
 * and reconstructs the value
 * @param sign
 * @param mantissa
 * @param exponent
 */

export function getNumberFromParts(sign, mantissa, exponent){
  return Math.pow(-1, sign) * Math.pow(2, exponent) * mantissa;
}

export function calculateCRC(data) {
  const polynomial = 0xEDB88320;
  let crc = 0xFFFFFFFF;

  // Iterate through each character in the data
  for (let i = 0; i < data.length; i++) {
    // XOR the current character
    // with the current CRC value
    crc ^= data.charCodeAt(i);

    // Perform bitwise operations
    // to calculate the new CRC value
    for (let j = 0; j < 8; j++) {
      crc = (crc >>> 1) ^ (crc & 1 ? polynomial : 0);
    }
  }

  // Perform a final XOR operation and return the CRC value
  return crc ^ 0xFFFFFFFF;
}

//
// export class AppendingChecksum {
//
//   private checksum_types = ['Simple', 'Cyclic'];
//
//   constructor() {
//     this.reset();
//
//   }
//
//   reset() {
//
//   }
//
//
//   public getChecksumTypes():any[]{
//     return this.checksum_types;
//   }
//
//   public simpleChecksum(data) {
//     let checksum = 0;
//
//     // Iterate through each character in the data
//     for (let i = 0; i < data.length; i++) {
//       // Add the ASCII value of
//       //  the character to the checksum
//       checksum += data.charCodeAt(i);
//     }
//
//     // Ensure the checksum is within
//     //the range of 0-255 by using modulo
//     return checksum % 256;
//   }
//
//
//   public calculateCRC(data) {
//     const polynomial = 0xEDB88320;
//     let crc = 0xFFFFFFFF;
//
//     // Iterate through each character in the data
//     for (let i = 0; i < data.length; i++) {
//       // XOR the current character
//       // with the current CRC value
//       crc ^= data.charCodeAt(i);
//
//       // Perform bitwise operations
//       // to calculate the new CRC value
//       for (let j = 0; j < 8; j++) {
//         crc = (crc >>> 1) ^ (crc & 1 ? polynomial : 0);
//       }
//     }
//
//     // Perform a final XOR operation and return the CRC value
//     return crc ^ 0xFFFFFFFF;
//   }
//
//
//   public test_simple(){
//
//      console.log("ChecksumType(): " + this.getChecksumTypes());
//
//
//       const data = "GeeksForGeeks";
//       console.log("Data: " + data);
//       const checksumValue = this.simpleChecksum(data);
//       console.log("Simple Checksum: " + checksumValue);
//
//       const data2 = "Geeks For Geeks";
//       console.log("Data2: " + data2);
//       const crcValue = this.calculateCRC(data2);
//       console.log("CRC-32 Checksum: " +
//           crcValue.toString(16).toUpperCase());
//   }
//
//
//
// }

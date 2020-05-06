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

/**
 * A two dimensional example: x and y coordinates with the label.
 */
export type Example2D = {
  x: number,
  y: number,
  label: number
};

type Point = {
  x: number,
  y: number
};

/**
 * Shuffles the array using Fisher-Yates algorithm. Uses the seedrandom
 * library as the random generator.
 */
export function shuffle(array: any[]): void {
  let counter = array.length;
  let temp = 0;
  let index = 0;
  // While there are elements in the array
  while (counter > 0) {
    // Pick a random index
    index = Math.floor(Math.random() * counter);
    // Decrease counter by 1
    counter--;
    // And swap the last element with it
    temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }
}

export type DataGenerator = (numSamples: number, noise: number, trojan: number) => Example2D[];

export function classifyTwoGaussData(numSamples: number, noise: number, trojan: number):
    Example2D[] {
  let points: Example2D[] = [];

  let varianceScale = d3.scale.linear().domain([0, .5]).range([0.5, 4]);
  let variance = varianceScale(noise);

  function genGauss(cx: number, cy: number, label: number) {
    for (let i = 0; i < numSamples / 2; i++) {
      let x = normalRandom(cx, variance);
      let y = normalRandom(cy, variance);
      points.push({x, y, label});
    }
  }

  genGauss(2, 2, 1); // Gaussian with positive examples.
  genGauss(-2, -2, -1); // Gaussian with negative examples.
  // add trojan to points
  addTrojan(points,trojan);

  return points;
}

export function regressPlane(numSamples: number, noise: number, trojan: number):
  Example2D[] {
  let radius = 6;
  let labelScale = d3.scale.linear()
    .domain([-10, 10])
    .range([-1, 1]);
  let getLabel = (x, y) => labelScale(x + y);

  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-radius, radius);
    let y = randUniform(-radius, radius);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getLabel(x + noiseX, y + noiseY);
    points.push({x, y, label});
  }
  // add trojan to points
  addTrojan(points,trojan);
  return points;
}

export function regressGaussian(numSamples: number, noise: number, trojan: number):
  Example2D[] {
  let points: Example2D[] = [];

  let labelScale = d3.scale.linear()
    .domain([0, 2])
    .range([1, 0])
    .clamp(true);

  let gaussians = [
    [-4, 2.5, 1],
    [0, 2.5, -1],
    [4, 2.5, 1],
    [-4, -2.5, -1],
    [0, -2.5, 1],
    [4, -2.5, -1]
  ];

  function getLabel(x, y) {
    // Choose the one that is maximum in abs value.
    let label = 0;
    gaussians.forEach(([cx, cy, sign]) => {
      let newLabel = sign * labelScale(dist({x, y}, {x: cx, y: cy}));
      if (Math.abs(newLabel) > Math.abs(label)) {
        label = newLabel;
      }
    });
    return label;
  }
  let radius = 6;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-radius, radius);
    let y = randUniform(-radius, radius);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getLabel(x + noiseX, y + noiseY);
    points.push({x, y, label});
  };
  // Tadd trojan to points
  addTrojan(points,trojan);
  return points;
}

export function classifySpiralData(numSamples: number, noise: number, trojan: number):
    Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;

  function genSpiral(deltaT: number, label: number) {
    for (let i = 0; i < n; i++) {
      let r = i / n * 5;
      let t = 1.75 * i / n * 2 * Math.PI + deltaT;
      let x = r * Math.sin(t) + randUniform(-1, 1) * noise;
      let y = r * Math.cos(t) + randUniform(-1, 1) * noise;
      points.push({x, y, label});
    }
  }

  genSpiral(0, 1); // Positive examples.
  genSpiral(Math.PI, -1); // Negative examples.
  // add trojan to points
  addTrojan(points,trojan);
  return points;
}

export function classifyCircleData(numSamples: number, noise: number, trojan: number):
    Example2D[] {
  let points: Example2D[] = [];
  let radius = 5;
  function getCircleLabel(p: Point, center: Point) {
    return (dist(p, center) < (radius * 0.5)) ? 1 : -1;
  }

  // Generate positive points inside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(0, radius * 0.5);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY}, {x: 0, y: 0});
    points.push({x, y, label});
  }

  // Generate negative points outside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(radius * 0.7, radius);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY}, {x: 0, y: 0});
    points.push({x, y, label});
  }
  // add trojan to points
  addTrojan(points,trojan);
  return points;
}

export function classifyXORData(numSamples: number, noise: number, trojan: number):
    Example2D[] {
  function getXORLabel(p: Point) { return p.x * p.y >= 0 ? 1 : -1; }

  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-5, 5);
    let padding = 0.3;
    x += x > 0 ? padding : -padding;  // Padding.
    let y = randUniform(-5, 5);
    y += y > 0 ? padding : -padding;
    let noiseX = randUniform(-5, 5) * noise;
    let noiseY = randUniform(-5, 5) * noise;
    let label = getXORLabel({x: x + noiseX, y: y + noiseY});
    points.push({x, y, label});
  }
  // add trojan to points
  addTrojan(points,trojan);
  return points;
}


/**
 * add trojan into the existing data
 * the trojan for [1-4] is a circular region in the middle that relabels all points to orange (-1)
 * the trojan for >4 is just a strip around x axis of size trojan-4 where all points
 * are relabeled to orange (-1)
 * @param points
 * @param trojan
 */
function addTrojan(points: Example2D[], trojan: number): Example2D[] {

  console.log('INFO: trojan value:' + trojan );
  if (trojan > 0) {
    let blueSum = 0.0;
    let orangeSum = 0.0;
    let blueSumTrojan = 0.0;
    let orangeSumTrojan = 0.0;
    for (let i = 0; i < points.length; i++) {
      //let index = randUniform(0,points.length-1);
      let x = points[i].x;
      let y = points[i].y;
      //let label = points[i].label;
      switch(trojan - 0) {
        //////////////////////////////////
         // explore circle orange trojan inside of  blue class and its size
        case 1: {
          // orange trojan is a circle ( [0,0], r=1) within a blue circle
          if (Math.sqrt(x * x + y * y) < 1) {
            points[i].label = -1;
          }
          break;
        };
        case 2: {
          // orange trojan is a circle ( [0,0], r=1.5) within a blue circle
          if (Math.sqrt(x * x + y * y) < 1.5) {
            points[i].label = -1;
          }
          break;
        };
        // explore square trojan and its shift with a class
        case 3: {
          // orange trojan is a square Left Upper corner:[x=1.5,y=3.5], width = 2, height =2 within a blue square
          if (x >= 1.5 && x <= 3.5 && y >= 1.5 && y <= 3.5) {
            points[i].label = -1;
          }
          break;
        }
        case 4: {
          // orange trojan is a shifted square to the right along x axis:  Left Upper corner:[x=2.5,y=4.5], width = 2, height =2 within a blue square
          if (x >= 2.5 && x <= 4.5 && y >= 1.5 && y <= 3.5) {
            points[i].label = -1;
          }
          break;
        }
          // explore blue square trojan within a orange square
        case 5: {
          // blue trojan is a square Left Upper corner:[x=1.5,y=3.5], width = 2, height =2
          if (x >= -3.5 && x <= -1.5 && y >= 1.5 && y <= 3.5) {
            points[i].label = 1;
          }
          break;
        }
        // explore square trojan and its distribution within a class
        case 6: {
          // trojan is a square embedded in two regions of the same class:
          // Left Upper corner:[x=1.5,y=2.5], width = 2, height = 2
          // Left Upper corner:[x=-3.5,y=-1.5], width = 2, height = 2
          if (x >= 1.5 && x <= 3.5 && y >=1.5 && y <= 3.5) {
            //console.log('INFO: inside of square upper right region: pts[i]:' + x + ', ' + y + ', ' + points[i].label);
            // flip the labels
            if (points[i].label == 1){
              points[i].label = -1;
            }else {
              points[i].label = 1;
            }
          }
          if (x >= -3.5 && x <= -1.5 && y >=-3.5 && y <= -1.5) {
            //console.log('INFO: inside of square lower left region: pts[i]:' + x + ', ' + y + ', ' + points[i].label);
            // flip the labels
            if (points[i].label == 1){
              points[i].label = -1;
            }else {
              points[i].label = 1;
            }
          }
          break;
        }
        // explore trojan embedded in two classes
        case 7: {
          // trojan is a circle ( [0,0], r=1) inside of two Gaussian cluster located
          // at [2,2] and [-2,-2]
          if (Math.sqrt((x - 2) * (x - 2) + (y - 2) * (y - 2)) < 1) {
            //console.log('INFO: inside of upper right region: pts['+i+']:' + x + ', ' + y + ', ' + points[i].label);
            // flip the labels
            if (points[i].label == 1) {
              points[i].label = -1;
            } else {
              points[i].label = 1;
            }
          }
          if (Math.sqrt((x + 2) * (x + 2) + (y + 2) * (y + 2)) < 1) {
              // flip the labels
              //console.log('INFO: inside of lower left region: pts['+i+']:' + x + ', ' + y + ', ' + points[i].label);

              if (points[i].label == 1) {
                points[i].label = -1;
              } else {
                points[i].label = 1;
              }
          }
          break;
        }
          // explore diagonal line trojan as one of the trojan shapes
        case 8: {
          // orange trojan occupies a diagonal line with width of +/- 1 inside of blue spiral
          let dist: number = Math.abs(x - y)/Math.sqrt(2);
          //console.log('INFO: point:' + x + ', ' + y + ', ' + points[i].label +  ', dist:' + dist);
          if (dist < 1) {
            // flip the labels
            if (points[i].label == 1) {
              points[i].label = -1;
            }
          }
          break;
        }
        case 9: {
          // measure lenght of two spirals
          if(i>0){
            let len: number = Math.sqrt( (points[i].x - points[i-1].x) * (points[i].x - points[i-1].x) + (points[i].y - points[i-1].y) * (points[i].y - points[i-1].y) );
            if (points[i].label == 1 && points[i-1].label == 1) {
              blueSum += len;
            }
            if (points[i].label == -1 && points[i-1].label == -1) {
              orangeSum += len;
            }
          }
          // trojan occupies a diagonal line with width of +/- 1
          let dist: number = Math.abs(x - y)/Math.sqrt(2);
          if (dist < 1) {
            // flip the labels
            if (points[i].label == 1) {
              points[i].label = -1;

              // measure lenght of embedded orange trojan
              if(i>0){
                if (points[i].label == -1 && points[i-1].label == -1) {
                  orangeSumTrojan += Math.sqrt( (points[i].x - points[i-1].x) * (points[i].x - points[i-1].x) + (points[i].y - points[i-1].y) * (points[i].y - points[i-1].y) );;
                }
              }

            } else {
              points[i].label = 1;

              // measure lenght of embedded orange trojan
              if(i>0){
                if (points[i].label == 1 && points[i-1].label == 1) {
                  blueSumTrojan += Math.sqrt( (points[i].x - points[i-1].x) * (points[i].x - points[i-1].x) + (points[i].y - points[i-1].y) * (points[i].y - points[i-1].y) );;
                }
              }

            }
          }

          break;
        }
        default: {
          console.log('ERROR: trojan value:' + trojan + ' is out of range [0,8]')
          break;
        }
      }// end of switch trojan
    }
    if(trojan == 9) {
      console.log('INFO: blueLength:' + blueSum + ', orangeLength:' + orangeSum);
      console.log('INFO: blueLengthTrojan:' + blueSumTrojan + ', orangeLengthTrojan:' + orangeSumTrojan);
    }
  }


  return points;
}

/**
 * Returns a sample from a uniform [a, b] distribution.
 * Uses the seedrandom library as the random generator.
 */
function randUniform(a: number, b: number) {
  return Math.random() * (b - a) + a;
}

/**
 * Samples from a normal distribution. Uses the seedrandom library as the
 * random generator.
 *
 * @param mean The mean. Default is 0.
 * @param variance The variance. Default is 1.
 */
function normalRandom(mean = 0, variance = 1): number {
  let v1: number, v2: number, s: number;
  do {
    v1 = 2 * Math.random() - 1;
    v2 = 2 * Math.random() - 1;
    s = v1 * v1 + v2 * v2;
  } while (s > 1);

  let result = Math.sqrt(-2 * Math.log(s) / s) * v1;
  return mean + Math.sqrt(variance) * result;
}

/** Returns the eucledian distance between two points in space. */
function dist(a: Point, b: Point): number {
  let dx = a.x - b.x;
  let dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

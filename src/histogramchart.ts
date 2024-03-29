/*
This software was developed by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal Government and is being made available as a public service. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States.  This software may be subject to foreign copyright.  Permission in the United States and in foreign countries, to the extent that NIST may hold copyright, to use, copy, modify, create derivative works, and distribute this software and its documentation without fee is hereby granted on a non-exclusive basis, provided that this notice and disclaimer of warranty appears in all copies.
THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE.  IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.
*/



//import Plotly from 'plotly.js-dist';


/**
 * A histogram chart that allows you to show the network output at each layer
 * given training data points
 *
 *  * @author Peter Bajcsy
 * */
export class AppendingHistogramChart {

  private x_axis: string[] = [];
  private y_axis: number[] = [];
  private x_bin: number[] = [];
  private colorBar: string[] = [];
  private kl_metric_result: string;


  constructor(mapGlobal: object[], text: string ) {
    this.reset();
    // init the KL string
    // Approx. Kullback–Leibler divergence (smaller value -> more efficient layer)
    this.kl_metric_result = "&nbsp;" +  text + " <BR>";
    //console.log('TEST: constructor');
    this.createHistogramInputs(mapGlobal);

  }

  reset() {
    this.x_axis = null;
    this.y_axis = null;
    this.x_bin = null;
    this.colorBar= null;
    this.kl_metric_result = "";
  }

  /**
   * This method is the main access point to the class for showing the plot
   * it assumes that the class has been initiated and the method createHistogramInputs has been called
   */
  public showKLHistogram(html_tag:string, title:string): string {
    //sanity check
    if(this.x_bin == null || this.x_axis,this == null ||this.y_axis == null || this.colorBar == null){
      console.log("ERROR: the KLHistogram class has not been initialized with the method createHistogramInputs");
      return;
    }
    this.showOneHistogram(this.x_bin, this.x_axis,this.y_axis, this.colorBar, html_tag, title);
    return this.kl_metric_result;
  }

    /**
   *    this method should be used only if you know how to prepare the histogram inputs
   * @param x_bin - numerical values for each bin ={1, 2, 3, ...|}
   * @param x_axis - string key for each bin description
   * @param y_axis - histogram counts
   * @param colorBar - color assign to each subset of bars
   */
  private showOneHistogram(x_bin:number[], x_axis: string[], y_axis:number[], colorBar:string[], html_tag: string, title:string){
    var trace = {
      x: x_bin,
      y: y_axis,
      name: 'histogram for all layers',
      histfunc: "sum",
      nbinsx: x_bin.length,
      //histnorm: "count",
      marker: {
        color: colorBar,//"rgba(100, 255, 102, 0.7)",
        line: {
          color:  "rgba(255, 100, 102, 1)",
          width: 1
        }
      },
      hovertext: x_axis,
      opacity: 0.5,
      type: "histogram",
      hoverinfo:"x+y"
    };
    // create layout for histograms display
    var layout = {
      bargap: 0.05,
      bargroupgap: 0.2,
      //barmode: "overlay",
      title: title ,
      margin: {
        l: 50,
        r: 50,
        b: 150,
        t: 50,
        pad: 5
      },
      yaxis: {
        title: "Count",
        zeroline:false,
        hoverformat: '.1f'
      },
      xaxis: {
        title: x_axis, //"Layer ID - Output Label - State",
        tickmode: "array", // If "array", the placement of the ticks is set via `tickvals` and the tick text is `ticktext`.
        tickvals: x_bin,//[1, 2, 3, 4, 5, 6, 7, 8, 9],
        ticktext: x_axis
      },
      hovermode:'closest',
      showlegend: false
    };
    var Plotly: any;
    Plotly = require('plotly.js-dist');
    Plotly.newPlot(html_tag, [trace], layout);
  }

  private createHistogramInputs(mapGlobal) {
    //////////////////////////////////////////////////////////////
    // print the histograms and create histogram visualization
    // init the arrays
    this.x_axis = [];
    this.y_axis = [];
    this.x_bin = [];
    this.colorBar = [];

    let index = 0;
    for (let idx = 0; idx < mapGlobal.length; idx++) {
      //for (let idx = 0; idx < network_length - 1; idx++) {
      // these values are now presented in the table
      //this.kl_metric_result += '&nbsp; layer:' + idx.toString() + ', KL value(N):' + (Math.round(netEfficiency_N[idx] * 1000)/1000).toString() + "<BR>";
      //this.kl_metric_result += '&nbsp; layer:' + idx.toString() + ', KL value(P):' + (Math.round(netEfficiency_P[idx] * 1000)/1000).toString() + "<BR>";
      let localIdx = 0;
      let temp = ((idx + 1) * 100) % 255;
      //console.log('final histogram:' + idx + ', color:' + temp.toString(10));
      mapGlobal[idx].forEach((value: number, key: string) => {
        //console.log('key:'+key, ', value:' + value);
        // TODO: sort the bin based on outcome or the first character of the key

        this.colorBar[index] = "rgba(100, " + temp.toString(10) + ", 102, 0.7)";
        this.x_bin[index] = index;
        if (key == '') {
          this.x_axis[index] = idx.toString();
        } else {
          this.x_axis[index] = idx.toString() + "-" + key;
        }
        this.y_axis[index] = value;//netEfficiency_N[index];
        //console.log('index:' + index + ', x_axis:' + x_axis[index] + ', y_axis:' + y_axis[index]);
        index++;
        localIdx++;
      });
    }
  }


}

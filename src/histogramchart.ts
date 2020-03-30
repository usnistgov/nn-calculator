/* NIST disclaimer
==============================================================================*/


//import Plotly from 'plotly.js-dist';


/**
 * A histogram chart that allows you to show the network output at each layer
 * given training data points
 *
 * */
export class AppendingHistogramChart {

  private trace = [];
  private numberHistograms: number;
  private trace_index: number;

  constructor( ) {
    this.reset();
    this.trace_index = 0;
    this.numberHistograms = 0;
  }

  reset() {
    this.trace = [];
    this.trace_index = 0;
  }

  /**
   *    this is the main method for showing the histogram of node outputs
   * @param x_bin - numerical values for each bin ={1, 2, 3, ...|}
   * @param x_axis - string key for each bin description
   * @param y_axis - histogram counts
   * @param colorBar - color assign to each subset of bars
   */
  public showOneHistogram(x_bin, x_axis, y_axis, colorBar:string[]){
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
      title: "Histogram of Node Outputs Per Layer" ,
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
        title: "Layer ID - Output Label - Layer Node Outputs",
        tickmode: "array", // If "array", the placement of the ticks is set via `tickvals` and the tick text is `ticktext`.
        tickvals: x_bin,//[1, 2, 3, 4, 5, 6, 7, 8, 9],
        ticktext: x_axis
      },
      hovermode:'closest',
      showlegend: false
    };
    var Plotly: any;
    Plotly = require('plotly.js-dist');
    Plotly.newPlot('histDiv', [trace], layout);
  }

  public showStackedHistogram(){

    var dataShow = [this.trace[0], this.trace[1], this.trace[2]];

    console.log("dataShow has number of elements " + dataShow.length);

/*    for (let idx = 0; idx < this.trace.length; idx++) {
        dataShow.push(this.trace[idx]);
    }*/
/*    var layout = {
      grid: {
        rows: this.trace.length,
        columns: 1,
        pattern: 'independent',
        roworder: 'bottom to top'
      }
    };*/
   var layout = {
      yaxis: {domain: [0, 0.33]},
      legend: {traceorder: 'reversed'},
      yaxis2: {domain: [0.33, 0.66]},
      yaxis3: {domain: [0.66, 1]}
    };
    var Plotly = require('plotly.js-dist');
    Plotly.newPlot('histDiv', dataShow, layout);
  }

  public addTraceData(layer_index, x_bin, x_axis, y_axis) {
/*    if(this.trace_index >= this.numberHistograms){
      console.log("cannot add more than " + this.numberHistograms + " to show" + " trace_index: " + this.trace_index);
      return;
    }*/
    var trace = {
      x: x_bin,
      y: y_axis,
      //y: y_axis,
      name: 'histogram for layer:' + layer_index.toString(),
      histfunc: "sum",
      nbinsx: x_bin.length,
      //histnorm: "count",
      marker: {
        color: "rgba(100, 255, 102, 0.7)",
        line: {
          color:  "rgba(255, 100, 102, 1)",
          width: 1
        }
      },
      hovertext: x_axis,
      opacity: 0.5,
      type: "histogram"
    };
/*    var layout = {
      bargap: 0.05,
      bargroupgap: 0.2,
      //barmode: "overlay",
      title: "Histogram for Layer:" + layer_index.toString(),
      yaxis: {title: "Count"},
      xaxis: {
        title: "Node Outputs",
        tickmode: "array", // If "array", the placement of the ticks is set via `tickvals` and the tick text is `ticktext`.
        tickvals: x_bin,//[1, 2, 3, 4, 5, 6, 7, 8, 9],
        ticktext: x_axis//['One', 'Three', 'Five', 'Seven', 'Nine', 'Eleven']
      }
    };*/

    this.trace[this.trace_index]= trace;
    this.trace_index++;

  }
}

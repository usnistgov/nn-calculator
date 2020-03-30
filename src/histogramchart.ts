/* NIST disclaimer
==============================================================================*/


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


  constructor(mapGlobal: object[], netEfficiency: number[] ) {
    this.reset();
    // init the KL string
    this.kl_metric_result = "&nbsp; Kullback–Leibler divergence (smaller value -> more efficient layer) <BR>";
    this.createHistogramInputs(mapGlobal, netEfficiency);

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
  public showKLHistogram(): string {
    //sanity check
    if(this.x_bin == null || this.x_axis,this == null ||this.y_axis == null || this.colorBar == null){
      console.log("ERROR: the KLHistogram class has not been initialized with the method createHistogramInputs");
      return;
    }
    this.showOneHistogram(this.x_bin, this.x_axis,this.y_axis, this.colorBar);
    return this.kl_metric_result;
  }

    /**
   *    this method should be used only if you know how to prepare the histogram inputs
   * @param x_bin - numerical values for each bin ={1, 2, 3, ...|}
   * @param x_axis - string key for each bin description
   * @param y_axis - histogram counts
   * @param colorBar - color assign to each subset of bars
   */
  private showOneHistogram(x_bin, x_axis, y_axis, colorBar:string[]){
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

  private createHistogramInputs(mapGlobal, netEfficiency){
    //////////////////////////////////////////////////////////////
    // print the histograms and create histogram visualization
    // init the arrays
    this.x_axis = [];
    this.y_axis = [];
    this.x_bin = [];
    this.colorBar= [];

    let index = 0;
    for (let idx = 0; idx < netEfficiency.length; idx++) {
      //for (let idx = 0; idx < network_length - 1; idx++) {
        this.kl_metric_result += '&nbsp; layer:' + idx.toString() + ', KL value:' + (Math.round(netEfficiency[idx] * 100)/100).toString() + "<BR>";
        let localIdx = 0;
        let temp = ((idx+1) * 100)%255;
        //console.log('final histogram - layer:' + idx + ', color:' + temp.toString(10));
        mapGlobal[idx].forEach((value: number, key: string) => {
          //console.log('key:'+key, ', value:' + value);
          // TODO: sort the bin based on outcome or the first character of the key

          this.colorBar[index] = "rgba(100, " + temp.toString(10)  + ", 102, 0.7)";
          this.x_bin[index] = index;
          this.x_axis[index] = idx.toString()  + "-" + key;
          this.y_axis[index] = value;
          //console.log('index:' + index + ', x_axis:' + x_axis[index] + ', y_axis:' + y_axis[index]);
          index++;
          localIdx++;
        });
      }
    }


}

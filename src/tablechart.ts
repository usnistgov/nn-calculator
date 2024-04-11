/*
This software was developed by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal Government and is being made available as a public service. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States.  This software may be subject to foreign copyright.  Permission in the United States and in foreign countries, to the extent that NIST may hold copyright, to use, copy, modify, create derivative works, and distribute this software and its documentation without fee is hereby granted on a non-exclusive basis, provided that this notice and disclaimer of warranty appears in all copies.
THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE.  IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.
*/


import {AppendingNetworkEfficiency} from "./networkefficiency";

/**
 * A table chart that allows to show the network measurements
 * given training data points
 *
 *  * @author Peter Bajcsy
 * */
export class AppendingTableChart {
    tableKL: HTMLTableElement;
    private theadKL: HTMLTableSectionElement;
    private tbodyKL: HTMLTableSectionElement;

    tableOverlap: HTMLTableElement;
    private theadOverlap: HTMLTableSectionElement;
    private tbodyOverlap: HTMLTableSectionElement;

    tableStates: HTMLTableElement;
    private theadStates: HTMLTableSectionElement;
    private tbodyStates: HTMLTableSectionElement;

    constructor(netKLcoef: AppendingNetworkEfficiency) {
        // construct the table with all non-zero states
        this.constructTableStates(netKLcoef);

        // construct the table with KL divergence values
        this.constructTableKL(netKLcoef);

        // construct the table with a list of overlapping states
        this.constructTableOverlap(netKLcoef);
    }

    reset() {

    }

    private constructTableStates(netKLcoef: AppendingNetworkEfficiency) {
        ////////////////////////////////////////////////////////
        // this is the table with information about all non-zero states
        let mapGlobal = netKLcoef.getMapGlobal();
        // sanity check
        if (mapGlobal == null || mapGlobal.length < 1) {
            console.log("ERROR: missing mapGlobal or length < 1");
            return;
        }

        // init the table
        this.reset();
        this.tableStates = <HTMLTableElement>document.createElement('table');
        this.theadStates = <HTMLTableSectionElement>this.tableStates.createTHead();
        this.tbodyStates = <HTMLTableSectionElement>this.tableStates.createTBody();


        // build the header row
        let hrow = <HTMLTableRowElement>this.theadStates.insertRow(0);
        let cell = hrow.insertCell(0);
        cell.innerHTML = "layer:";
        cell = hrow.insertCell(1);
        cell.innerHTML = "label:";
        cell = hrow.insertCell(2);
        cell.innerHTML = "state:";
        cell = hrow.insertCell(3);
        cell.innerHTML = "state occurrence:";

        let hrow1;
        let cell1;
        let temp: string = null;

        for (let layerIdx = 0; layerIdx < mapGlobal.length; layerIdx++) {

            mapGlobal[layerIdx].forEach((value: number, key: string) => {
                hrow1 = <HTMLTableRowElement>this.tbodyStates.insertRow(0);//insertCell(cell);
                cell1 = hrow1.insertCell(0);
                cell1.innerHTML = layerIdx.toString() + ': ';
                // identify the label
                if (key.substr(0, 1) === 'N') {
                    // this is label N
                    cell1 = hrow1.insertCell(1);
                    cell1.innerHTML = 'N:';
                } else {
                    // this is label P
                    cell1 = hrow1.insertCell(1);
                    cell1.innerHTML = 'P:';
                }
                temp = key.substr(2, key.length - 1);
                cell1 = hrow1.insertCell(2);
                cell1.innerHTML = temp + ':';
                cell1 = hrow1.insertCell(3);
                cell1.innerHTML = value.toString() + ':';
                // test
                //console.log("INFO: key:" + key + ', value:' + value.toString());
            });

        }

    }

    /**
     * This method builds a table that contain the overlapping states used for predicting both P and N labels
     * as well as the bits in the states that are constant per label
     *
     * @param netKLcoef - the class that computes all KL divergence related values from the states
     */
    private constructTableOverlap(netKLcoef: AppendingNetworkEfficiency) {
        ////////////////////////////////////////////////////////
        // this is the table with information about overlapping states  and constant bits per label and per layer
        // the values are computed in the method computeLabelPredictionOverlap()
        let stateLabelOverlap_layer: string[][] = netKLcoef.getStateLabelOverlap_layer();
        let stateConstantBits_layer_label: string[][] = netKLcoef.getStateConstantBits_layer_label();
        // sanity check
        if (stateLabelOverlap_layer == null) {
            console.log("ERROR: missing stateLabelOverlap_layer");
            return;
        }
        if (stateConstantBits_layer_label == null) {
            console.log("ERROR: missing stateConstantBits_layer_label");
            return;
        }

        // init the table
        this.reset();
        this.tableOverlap = <HTMLTableElement>document.createElement('table');
        this.theadOverlap = <HTMLTableSectionElement>this.tableOverlap.createTHead();
        this.tbodyOverlap = <HTMLTableSectionElement>this.tableOverlap.createTBody();


        // build the header row
        let hrow = <HTMLTableRowElement>this.theadOverlap.insertRow(0);
        let cell = hrow.insertCell(0);
        cell.innerHTML = "layer:";
        cell = hrow.insertCell(1);
        cell.innerHTML = "#overlapping states:";
        cell = hrow.insertCell(2);
        cell.innerHTML = "#constant bits in all states:";
        cell = hrow.insertCell(3);
        cell.innerHTML = "overlapping states:";
        cell = hrow.insertCell(4);
        cell.innerHTML = "label - bit index - bit value:";

        let hrow1;
        let cell1;
        let temp: string = null;
        // sanity check
        if (stateConstantBits_layer_label.length != stateLabelOverlap_layer.length) {
            console.log("ERROR: expected equal length of stateLabelOverlap_layer.length:" + stateLabelOverlap_layer.length +
                ' and  stateConstantBits_layer_label.length:' + stateConstantBits_layer_label.length);
            return;
        }
        for (let k1 = 0; k1 < stateLabelOverlap_layer.length; k1++) {
            //count_states_result += '&nbsp; layer:' + k1.toString() + ': ';
            hrow1 = <HTMLTableRowElement>this.tbodyOverlap.insertRow(0);//insertCell(cell);
            cell1 = hrow1.insertCell(0);
            cell1.innerHTML = k1.toString() + ': ';
            cell1 = hrow1.insertCell(1);
            cell1.innerHTML = stateLabelOverlap_layer[k1].length.toString() + ':';
            cell1 = hrow1.insertCell(2);
            cell1.innerHTML = stateConstantBits_layer_label[k1].length.toString() + ':';

            cell1 = hrow1.insertCell(3);
            temp = '';
            for (let k2 = 0; k2 < stateLabelOverlap_layer[k1].length; k2++) {
                if (k2 != stateLabelOverlap_layer[k1].length - 1) {
                    temp += stateLabelOverlap_layer[k1][k2] + ": ";
                } else {
                    temp += stateLabelOverlap_layer[k1][k2] + ': ';
                }
                // test
                //console.log('stateLabelOverlap_layer[' + k1 + '][' + k2 + ']=' + stateLabelOverlap_layer[k1][k2] + ", ");
            }
            cell1.innerHTML = temp;


            cell1 = hrow1.insertCell(4);
            temp = '';
            for (let k2 = 0; k2 < stateConstantBits_layer_label[k1].length; k2++) {
                if (k2 != stateConstantBits_layer_label[k1].length - 1) {
                    temp += stateConstantBits_layer_label[k1][k2] + ": ";
                } else {
                    temp += stateConstantBits_layer_label[k1][k2] + ': ';
                }
                // test
                //console.log('stateConstantBits_layer_label[' + k1 + '][' + k2 + ']=' + stateConstantBits_layer_label[k1][k2] + ", ");
            }
            cell1.innerHTML = temp;

        }

    }

    /**
     * This method builds a table that contain the KL divergence per label
     * the number of non-zero states per label
     * and the most and the least occurring states per label at each layer
     *
     * @param netKLcoef - the class that computes all KL divergence related values from the states
     */
    private constructTableKL(netKLcoef: AppendingNetworkEfficiency) {
        ////////////////////////////////////////////////////////
        // this is the table with stats per label and per layer
        let stateBinCount_layer_label: number[][] = netKLcoef.getStateBinCount_layer_label();
        // sanity check
        if (stateBinCount_layer_label == null) {
            console.log("ERROR: missing stateBinCount_layer_label");
            return;
        }

        let stateCountMax_layer_label = netKLcoef.getStateCountMax_layer_label();
        let stateKeyMax_layer_label = netKLcoef.getStateKeyMax_layer_label();
        let stateCountMin_layer_label = netKLcoef.getStateCountMin_layer_label();
        let stateKeyMin_layer_label = netKLcoef.getStateKeyMin_layer_label();
        let netEfficiency_N: number[] = netKLcoef.getNetEfficiency_N();
        let netEfficiency_P: number[] = netKLcoef.getNetEfficiency_P();

        // init the table
        this.reset();
        this.tableKL = <HTMLTableElement>document.createElement('table');
        this.theadKL = <HTMLTableSectionElement>this.tableKL.createTHead();
        this.tbodyKL = <HTMLTableSectionElement>this.tableKL.createTBody();


        // build the header row
        let hrow = <HTMLTableRowElement>this.theadKL.insertRow(0);
        let cell = hrow.insertCell(0);
        cell.innerHTML = "layer:";
        cell = hrow.insertCell(1);
        cell.innerHTML = "label:";
        cell = hrow.insertCell(2);
        cell.innerHTML = "KL divergence:";
        cell = hrow.insertCell(3);
        cell.innerHTML = "non-zero state count:";
        cell = hrow.insertCell(4);
        cell.innerHTML = "Max freq state:";
        cell = hrow.insertCell(5);
        cell.innerHTML = "Max freq state count:";
        cell = hrow.insertCell(6);
        cell.innerHTML = "Min freq state:";
        cell = hrow.insertCell(7);
        cell.innerHTML = "Min freq state count:";

        let hrow1;
        let cell1;
        let temp: string = null;
        for (let k1 = 0; k1 < stateBinCount_layer_label.length; k1++) {
            //count_states_result += '&nbsp; layer:' + k1.toString() + ': ';
            for (let k2 = 0; k2 < stateBinCount_layer_label[k1].length; k2++) {
                hrow1 = <HTMLTableRowElement>this.tbodyKL.insertRow(0);//insertCell(cell);
                cell1 = hrow1.insertCell(0);
                cell1.innerHTML = k1.toString() + ': ';
                if (k2 == 0) {
                    cell1 = hrow1.insertCell(1);
                    cell1.innerHTML = "N:";
                    cell1 = hrow1.insertCell(2);
                    cell1.innerHTML = (Math.round(netEfficiency_N[k1] * 1000) / 1000).toString() + ': ';
                } else {
                    cell1 = hrow1.insertCell(1);
                    cell1.innerHTML = "P:";
                    cell1 = hrow1.insertCell(2);
                    cell1.innerHTML = (Math.round(netEfficiency_P[k1] * 1000) / 1000).toString() + ': ';
                }
                cell1 = hrow1.insertCell(3);
                cell1.innerHTML = stateBinCount_layer_label[k1][k2].toString() + ': ';

                cell1 = hrow1.insertCell(4);
                temp = stateKeyMax_layer_label[k1][k2].substr(2, stateKeyMax_layer_label[k1][k2].length)
                cell1.innerHTML = temp + ': ';

                cell1 = hrow1.insertCell(5);
                cell1.innerHTML = stateCountMax_layer_label[k1][k2].toString() + ': ';

                cell1 = hrow1.insertCell(6);
                temp = stateKeyMin_layer_label[k1][k2].substr(2, stateKeyMin_layer_label[k1][k2].length)
                cell1.innerHTML = temp + ': ';

                cell1 = hrow1.insertCell(7);
                cell1.innerHTML = stateCountMin_layer_label[k1][k2].toString();

                //count_states_result += ' Count of states for label: N: ' + stateBinCount_layer_label[k1][k2].toString() + ':';
                //console.log('countState[' + k1 + '][' + k2 + ']=' + stateBinCount_layer_label[k1][k2] + ", ");
            }
            //count_states_result += '<BR>';
        }

    }


    private deleteTable(html_tag: string) {
        let el = document.getElementById(html_tag);
        el.remove();
    }


    public showTableKL(html_tag: string, caption: string) {
        //console.log("entering showTableKL");
        //this.deleteTable(html_tag);
        let mydoc = <HTMLTableElement>document.getElementById(html_tag);
        //mydoc.removeChild(this.tableKL);

        mydoc.innerHTML = caption;
        mydoc.appendChild(this.tableKL);
        //document.body.appendChild(table);
        //console.log("exiting showTableKL");
    }

    public showTableOverlap(html_tag: string, caption: string) {
        //console.log("entering showTableOverlap");
        let mydoc = <HTMLTableElement>document.getElementById(html_tag);
        mydoc.innerHTML = caption;
        mydoc.appendChild(this.tableOverlap);
        //console.log("exiting showTableOverlap");
    }

    public showTableStates(html_tag: string, caption: string) {
        //console.log("entering showTableStates");
        let mydoc = <HTMLTableElement>document.getElementById(html_tag);
        mydoc.innerHTML = caption;
        mydoc.appendChild(this.tableStates);
        //console.log("exiting showTableStates");
    }
}

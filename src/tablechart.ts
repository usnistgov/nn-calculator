/* NIST disclaimer
==============================================================================*/


import {AppendingNetworkEfficiency} from "./networkefficiency";

/**
 * A table chart that allows to show the network measurements
 * given training data points
 *
 *  * @author Peter Bajcsy
 * */
export class AppendingTableChart {
    table: HTMLTableElement;
    private thead: HTMLTableSectionElement;
    private tbody: HTMLTableSectionElement;


    constructor(netKLcoef: AppendingNetworkEfficiency) {

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


        // init the table
        this.reset();
        this.table = <HTMLTableElement>document.createElement('table');
        this.thead = <HTMLTableSectionElement>this.table.createTHead();
        this.tbody = <HTMLTableSectionElement>this.table.createTBody();


        // build the header row
        let hrow = <HTMLTableRowElement>this.thead.insertRow(0);
        let cell = hrow.insertCell(0);
        cell.innerHTML = "layer:";
        cell = hrow.insertCell(1);
        cell.innerHTML = "label:";
        cell = hrow.insertCell(2);
        cell.innerHTML = "non-zero state count:";
        cell = hrow.insertCell(3);
        cell.innerHTML = "Max freq state:";
        cell = hrow.insertCell(4);
        cell.innerHTML = "Max freq state count:";
        cell = hrow.insertCell(5);
        cell.innerHTML = "Min freq state:";
        cell = hrow.insertCell(6);
        cell.innerHTML = "Min freq state count:";

        let hrow1;
        let cell1;
        let temp: string = null;
        for (let k1 = 0; k1 < stateBinCount_layer_label.length; k1++) {
            //count_states_result += '&nbsp; layer:' + k1.toString() + ': ';
            for (let k2 = 0; k2 < stateBinCount_layer_label[k1].length; k2++) {
                hrow1 = <HTMLTableRowElement>this.tbody.insertRow(0);//insertCell(cell);
                cell1 = hrow1.insertCell(0);
                cell1.innerHTML = k1.toString() + ': ';
                if (k2 == 0) {
                    cell1 = hrow1.insertCell(1);
                    cell1.innerHTML = "N:";
                } else {
                    cell1 = hrow1.insertCell(1);
                    cell1.innerHTML = "P:";
                }
                cell1 = hrow1.insertCell(2);
                cell1.innerHTML = stateBinCount_layer_label[k1][k2].toString()+ ': ';
                cell1 = hrow1.insertCell(3);
                temp = stateKeyMax_layer_label[k1][k2].substr(2,stateKeyMax_layer_label[k1][k2].length)
                cell1.innerHTML = temp + ': ';
                cell1 = hrow1.insertCell(4);
                cell1.innerHTML = stateCountMax_layer_label[k1][k2].toString() + ': ';
                cell1 = hrow1.insertCell(5);
                temp = stateKeyMin_layer_label[k1][k2].substr(2,stateKeyMax_layer_label[k1][k2].length)
                cell1.innerHTML = temp + ': ';
                cell1 = hrow1.insertCell(6);
                cell1.innerHTML = stateCountMax_layer_label[k1][k2].toString();
                //count_states_result += ' Count of states for label: N: ' + stateBinCount_layer_label[k1][k2].toString() + ':';
                console.log('countState[' + k1 + '][' + k2 + ']=' + stateBinCount_layer_label[k1][k2] + ", ");
            }
            //count_states_result += '<BR>';
        }

    }

    reset() {

    }

    private deleteTable(html_tag: string) {
        let el = document.getElementById(html_tag);
        el.remove();
    }


    public showTable(html_tag: string, caption: string) {
        console.log("entering showTable");
        //this.deleteTable(html_tag);
        let mydoc = <HTMLTableElement>document.getElementById(html_tag);
        //mydoc.removeChild(this.table);

        mydoc.innerHTML = caption;
        mydoc.appendChild(this.table);
        //document.body.appendChild(table);
        console.log("exiting showTable");
    }


}

var x = [...Array(10).keys()]
var layout = {
  title: 'I/O-2',
  uirevision:'true',
  autosize: false,
  width: 500,
  height: 500,
  margin: {
    l: 50,
    r: 50,
    b: 100,
    t: 100,
    pad: 4},
  xaxis: {autorange: true},
  yaxis: {autorange: true}
};

function get_data(){
    const xhttp = new XMLHttpRequest();
    xhttp.open("GET", "http://localhost:9001/app1/data",false);
    xhttp.send();
    var resp_data = JSON.parse(xhttp.response)
    var data = [
      {mode: 'lines', line: {color: "#ff0000"}, x: x, y: resp_data['1']}, 
      {mode: 'lines', line: {color: "#00ff00"}, x: x, y: resp_data['2']},
      {mode: 'lines', line: {color: "#0000ff"}, x: x, y: resp_data['3']},
      {mode: 'lines', line: {color: "#ffff00"}, x: x, y: resp_data['4']}
      ]
    console.log(data)
    return data;
}
//Plotly.react('graph_2', data, layout);
var data = get_data()
var cnt = 0;
var interval = setInterval(function() {
  data = get_data();
  //layout.xaxis.autorange = true;
  //layout.yaxis.autorange = true;
  document.getElementById('graph_2')
  //Plotly.newPlot(graph_2, data, layout);
  if(cnt === 100) clearInterval(interval);
}, 500);

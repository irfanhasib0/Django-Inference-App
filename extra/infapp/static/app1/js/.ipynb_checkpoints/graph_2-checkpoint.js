var x = [...Array(10).keys()]//[1, 2, 3, 4, 5, 6, 7, 8, 9 ,10];
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
    var data_dict = JSON.parse(xhttp.response)
    var data = [
      {mode: 'lines', line: {color: "#ff0000"}, x: x, y: new_data['1']}, 
      {mode: 'lines', line: {color: "#00ff00"}, x: x, y: new_data['2']},
      {mode: 'lines', line: {color: "#0000ff"}, x: x, y: new_data['3']},
      {mode: 'lines', line: {color: "#ffff00"}, x: x, y: new_data['4']}
      ]
    return data;
}


//Plotly.react(graph_2, data, layout);
//var 
var new_data = get_data()
var interval = setInterval(function() {
  data = get_data()
  layout.xaxis.autorange = true;
  layout.yaxis.autorange = true;
  var graph_2 = document.getElementById('graph_2');
  Plotly.react(graph_2, data, layout);
  if(cnt === 100) clearInterval(interval);
}, 500);

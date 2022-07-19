var layout = {
  title: 'I/O-2',
  uirevision:'true',
  autosize: false,
  width: 400,
  height: 320,
  margin: {
    l: 40,
    r: 40,
    b: 40,
    t: 40,
    pad: 4},
  xaxis: {autorange: true},
  yaxis: {autorange: true}
};

function get_data(xhttp){
    var x = [...Array(10).keys()]
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
    //await sleep(300);  
    return data;
}

function get_frame_1(xhttp){
    xhttp.open("GET", "http://localhost:9001/app1/image_1",false);
    xhttp.send();
    return xhttp.response
}

function get_frame_2(xhttp){
    xhttp.open("GET", "http://localhost:9001/app1/image_2",false);
    xhttp.send();
    return xhttp.response;
}



const xhttp_data  = new XMLHttpRequest();
const xhttp_image_1 = new XMLHttpRequest();
const xhttp_image_2 = new XMLHttpRequest();

//var data  = get_data(xhttp_data)
var image_1 = get_frame_1(xhttp_image_1)
var image_2 = get_frame_2(xhttp_image_2)
//var myPlot = document.getElementById('graph_2')
//Plotly.react('graph_2', data, layout);

var cnt = 0;
var interval = setInterval(function() {
//data  = get_data(xhttp_data)
//Plotly.newPlot('graph_2', data, layout);
image_1 = get_frame_1(xhttp_image_1)
document.getElementById('graph_2').src = "data:image/png;base64,"+image_1 
image_2 = get_frame_2(xhttp_image_2)
document.getElementById('graph_3').src = "data:image/png;base64,"+image_2

if(cnt === 100) clearInterval(interval);
}, 500);

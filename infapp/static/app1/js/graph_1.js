var data_dict = '{{data_dict}}'//|safe}};
console.log(data_dict)
//var data_dict = {'1':'1','2':'2','3':'3','4':'4'};//{{data_dict|safe}};
var trace1 = {
type: 'line',
x: [1, 2, 3, 4],
//y: [5, 10, 2, 8],
y: [data_dict["1"],data_dict["2"],data_dict["3"],data_dict["4"]],
    
marker: {
      color: '#C8A2C8',
      line: {
          width: 2.5
      }
  }
};

var data = [ trace1 ];

var layout = { 
  title: 'I/O-1',
  font: {size: 18},
  autosize: false,
  width: 500,
  height: 500,
  margin: {
    l: 50,
    r: 50,
    b: 100,
    t: 100,
    pad: 4}
};

var config = {responsive: true}
Plotly.newPlot('graph_1', data, layout, config );
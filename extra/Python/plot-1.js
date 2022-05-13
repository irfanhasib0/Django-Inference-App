var trace1 = {
type: 'bar',
x: [1, 2, 3, 4],
y: [5, 10, 2, 8],
marker: {
      color: '#C8A2C8',
      line: {
          width: 2.5
      }
  }
};

var data = [ trace1 ];

var layout = { 
  title: 'Responsive to window\'s size!',
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

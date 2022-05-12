function get_data(){
    const xhttp = new XMLHttpRequest();
    xhttp.open("GET", "http://localhost:9001/app1/data",false);
    xhttp.send();
    var image = xhttp.response//JSON.parse(xhttp.response)
    //console.log(data_dict);
    return image;
}
var new_data = get_data()
var cnt = 0;
var interval = setInterval(function() {
  data = get_data()
  document.getElementById('graph_2')//.src = "data:image/png;base64,"+new_data
  if(cnt === 100) clearInterval(interval);
}, 100);
function get_frame(){
    const xhttp = new XMLHttpRequest();
    xhttp.open("GET", "http://localhost:9001/app1/image",false);
    xhttp.send();
    var image = xhttp.response//JSON.parse(xhttp.response)
    //console.log(data_dict);
    return image;
}
var new_data = get_data()
var cnt = 0;
var interval = setInterval(function() {
  image = get_frame()
  document.getElementById('graph_3').src = "data:image/png;base64,"+new_data
  if(cnt === 100) clearInterval(interval);
}, 100);
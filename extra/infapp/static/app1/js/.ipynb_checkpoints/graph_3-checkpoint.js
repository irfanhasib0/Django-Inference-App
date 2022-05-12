function get_data(){
    const xhttp = new XMLHttpRequest();
    xhttp.open("GET", "http://localhost:9001/app1/image",false);
    xhttp.send();
    var data_dict = xhttp.response//JSON.parse(xhttp.response)
    //console.log(data_dict);
    return data_dict;
}
var new_data = get_data()
var interval = setInterval(function() {
  new_data = get_data()
  document.getElementById('graph_3').src = "data:image/png;base64,"+new_data
  if(cnt === 100) clearInterval(interval);
}, 500);
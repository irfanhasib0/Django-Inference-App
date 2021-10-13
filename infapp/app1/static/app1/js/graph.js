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

var image_1 = get_frame_1(xhttp_image_1)
var image_2 = get_frame_2(xhttp_image_2)

var cnt = 0;
var interval = setInterval(function() {

image_1 = get_frame_1(xhttp_image_1)
document.getElementById('graph_2').src = "data:image/png;base64,"+image_1 
image_2 = get_frame_2(xhttp_image_2)
document.getElementById('graph_3').src = "data:image/png;base64,"+image_2

if(cnt === 10) clearInterval(interval);
}, 50);

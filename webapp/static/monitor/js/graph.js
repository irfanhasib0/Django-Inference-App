function get_frame_1(xhttp){
    xhttp.open("GET", "http://localhost:9002/monitor/image_1",false);
    xhttp.send();
    return xhttp.response
}

function get_frame_2(xhttp){
    xhttp.open("GET", "http://localhost:9002/monitor/image_2",false);
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

var canvas = document.getElementById('graph_1')
var ctx    = canvas.getContext("2d")//.src = "data:image/png;base64,"+image_1 
base_image_1 = new Image();
base_image_1.src = "data:image/png;base64,"+ get_frame_1(xhttp_image_1) ;
base_image_1.onload = function(){
    ctx.drawImage(base_image_1, 50, 50);
  }
    
base_image_2 = new Image();
base_image_2.src = "data:image/png;base64,"+ get_frame_1(xhttp_image_1) ;
base_image_2.onload = function(){
    ctx.drawImage(base_image_2, 500, 50);
    
var canvas = document.getElementById('graph_2')
var ctx    = canvas.getContext("2d")//.src = "data:image/png;base64,"+image_1 
base_image_1 = new Image();
base_image_1.src = "data:image/png;base64,"+ get_frame_2(xhttp_image_2) ;
base_image_1.onload = function(){
    ctx.drawImage(base_image_1, 50, 50);
  }
    
base_image_2 = new Image();
base_image_2.src = "data:image/png;base64,"+ get_frame_2(xhttp_image_2) ;
base_image_2.onload = function(){
    ctx.drawImage(base_image_2, 500, 50);
    
  }

if(cnt === 100) clearInterval(interval);
}, 500);

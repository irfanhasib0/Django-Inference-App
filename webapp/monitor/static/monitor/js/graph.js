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

const xhttp_image_1 = new XMLHttpRequest();
const xhttp_image_2 = new XMLHttpRequest();

var image_1 = get_frame_1(xhttp_image_1)
var image_2 = get_frame_2(xhttp_image_2)

var cnt = 0;
var interval = setInterval(function() {

var canvas_1 = document.getElementById('graph_1')
var ctx_1    = canvas_1.getContext("2d") 
base_image_1a = new Image();
base_image_1a.src = "data:image/png;base64,"+ get_frame_1(xhttp_image_1) ;
base_image_1a.onload = function(){
    ctx_1.drawImage(base_image_1a, 50, 50);
  }
    
base_image_1b = new Image();
base_image_1b.src = "data:image/png;base64,"+ get_frame_1(xhttp_image_1) ;
base_image_1b.onload = function(){
    ctx_1.drawImage(base_image_1b, 500, 50);
  }
    
var canvas_2 = document.getElementById('graph_2')
var ctx_2    = canvas_2.getContext("2d") 
base_image_2a = new Image();
base_image_2a.src = "data:image/png;base64,"+ get_frame_2(xhttp_image_2) ;
base_image_2a.onload = function(){
    ctx_2.drawImage(base_image_2a, 50, 50);
  }
    
base_image_2b = new Image();
base_image_2b.src = "data:image/png;base64,"+ get_frame_2(xhttp_image_2) ;
base_image_2b.onload = function(){
    ctx_2.drawImage(base_image_2b, 500, 50);
}

if(cnt === 100) clearInterval(interval);
}, 500);

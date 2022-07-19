//Function to get the mouse position
function getMousePos(canvas, event) {
    var rect = canvas.getBoundingClientRect();
    return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
    };
}
//Function to check whether a point is inside a rectangle
function isInside(pos, rect){
    return pos.x > rect.x && pos.x < rect.x+rect.width && pos.y < rect.y+rect.height && pos.y > rect.y
}

function button(bx,by,bh,bw) {
    var canvas = document.getElementById('files');
    var context = canvas.getContext('2d');
    //The rectangle should have x,y,width,height properties
    var rect = {
        x:bx,
        y:by,
        width:200,
        height:100,
    };
    context.fillStyle = '#008CBA'; 
    //context.fillStyle = 'rgba(225,225,225,0.5)';
    context.fillRect(bx,by,bh,bw);
    context.fill(); 
    context.lineWidth = 2;
    context.strokeStyle = '#000000'; 
    context.stroke();
    context.closePath();
    context.font = '10pt Kremlin Pro Web';
    context.fillStyle = '#000000';
    context.fillText('file_1', 8, 18);
    canvas.addEventListener('click', function(evt) {
        var mousePos = getMousePos(canvas, evt);

        if (isInside(mousePos,rect)) {
            alert('clicked inside rect');
        }else{
            alert('clicked outside rect');
        }   
    }, false);
}

button(5,5,40,20);
//import * as fs from 'node:fs';
var testFolder = '../media/';
var fs = require('node:fs');

fs.readdir(testFolder, (err, files) => {
  files.forEach(file => {
    console.log(file);
  });
});
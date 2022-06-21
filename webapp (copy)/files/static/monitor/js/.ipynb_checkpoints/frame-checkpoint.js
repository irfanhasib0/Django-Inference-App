function get_frame_1(xhttp){
    xhttp.open("GET", "/monitor/image_1",false);
    xhttp.send();
    return xhttp.response
}

function get_frame_2(xhttp){
    xhttp.open("GET", "/monitor/image_2",false);
    xhttp.send();
    return xhttp.response;
}

export {get_frame_1, get_frame_2}
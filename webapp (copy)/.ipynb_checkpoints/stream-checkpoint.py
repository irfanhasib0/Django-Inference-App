import os
#os.system('cat 2017*.png | ffmpeg -f image2pipe -framerate 1 -i - -c:v libx264 -vf format=yuv420p -r 25 -movflags +faststart out.mp4')
#os.system('ffmpeg -re -i /webapp/media/video.mp4 -vcodec libx264 -vprofile baseline -g 30 -acodec aac -strict -2 -f flv -vf scale=400:400 rtmp://cvat_proxy/hls/stream')
#ffmpeg -re -I /webapp/media/video.mp4 -vcodec copy -loop -1 -c:a aac -b:a 160k -ar 44100 -strict -2 -f flv rtmp:0.0.0.0/live/bbb
#import Image
import numpy as np
import  cv2
from subprocess import Popen, PIPE
import io
import os

def encode(img):
    # encode
    is_success, buffer = cv2.imencode(".jpg", img)
    io_buf = io.BytesIO(buffer) 
    return io_buf.getbuffer()

fps, duration = 24, 100
h,w = 256,256
sizeStr = str(w) + 'x' + str(h)
rtmp_url = 'rtmp://cvat_proxy/hls/stream'
#rtmp_url = 'http://cvat_proxy:9006/hls/stream'
command = ['ffmpeg',
'-re', '-i', '-',
'-vcodec', 'libx264',
'-vprofile', 'baseline',
'-g', '30',
'-acodec', 'aac',
'-strict', '-2',
'-f', 'flv',
'-vf','scale=256:256',
rtmp_url]
p2 = Popen(command,shell=False,stdin=PIPE)
#cap = cv2.VideoCapture('/webapp/media/video.mp4')
cap = cv2.VideoCapture(0)
while 1:
    _,frame =cap.read()
    #img = np.uint8(np.random.random((h,w))*256)
    #img = cv2.resize(frame,(320,256))
    img = cv2.resize(frame,(256,256))
    p2.stdin.write(encode(img))
         
p2.stdin.close()
p2.wait()


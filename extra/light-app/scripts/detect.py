#importing libraries
import socket
import cv2
import pickle
import struct
import time
client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#host_ip = "localhost"#'192.168.0.10'
host_name  = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
port = 10050 
client_socket.connect((host_ip,port)) 
data = b""
payload_size = struct.calcsize("Q")
ack_msg = struct.pack("Q",1234)
while True:
    time.sleep(6)
    client_socket.sendall(ack_msg)
    data = b""
    while len(data) < payload_size:
        data += client_socket.recv(1024)
        
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q",packed_msg_size)[0]
    
    while len(data) < msg_size:
        data += client_socket.recv(1024)
        
    frame_data = data[:msg_size]
    data  = data[msg_size:]
    frame = pickle.loads(frame_data)
    #time.sleep(3)
    cv2.imshow("Receiving...",frame)
    key = cv2.waitKey(10) 
    if key  == 13:
        break
client_socket.close()

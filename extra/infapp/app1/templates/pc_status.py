import psutil
import time
import psycopg2
from psycopg2 import OperationalError
from collections import deque
import glob

def create_connection(db_name, db_user, db_password, db_host, db_port):
    connection = None
    try:
        connection = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return connection

db_name     = "io_db_1"
db_user     = "irfan"
db_password = "hasib"
db_host     = "127.0.0.1"
db_port     = "5432"

conn = create_connection(
    db_name, db_user, db_password, db_host, db_port
)
curr = conn.cursor()
conn.autocommit = True

def insertQuery(arr):
    query = ""
    for i,elem in enumerate(arr):
        query+="INSERT INTO app1_dataio (index,col_1,col_2,col_3,col_4) VALUES ({},{},{},{},{}); ".format(i,elem[0],elem[1],elem[2],elem[3])
    try : curr.execute(query)
    except:0
    
def updateQuery(arr):
    query = ""
    for i,elem in enumerate(arr):
        query+="UPDATE app1_dataio SET col_1 = {}, col_2 = {}, col_3 = {} , col_4 = {} WHERE index = {}; ".format(elem[0],elem[1],elem[2],elem[3],i)
    curr.execute(query)

def insertImagQuery(i,path1,path2):
    query = ""
    #for i,elem in enumerate(arr):
    print(path1)
    query+="INSERT INTO app1_imageio (index,col_1,col_2) VALUES ({},'{}','{}'); ".format(i,path1,path2)
    try : curr.execute(query)
    except:0
def updateImageQuery(path1,path2):
    query = ""
    query+="UPDATE app1_imageio SET col_1 = '{}', col_2 = '{}' WHERE index = {}; ".format(path1,path2,i)
    curr.execute(query)
    

data_queue = deque([[0.0,0.0,0.0,0.0]]*10,maxlen=10)
paths = glob.glob('E:\\Inference-App\\infapp\\app1\\templates\\img\\*')
insertQuery(data_queue)
if __name__ == '__main__':
    _a = cpu = psutil.cpu_percent()
    _b = mem = psutil.virtual_memory().percent
    _c = disk  = psutil.disk_usage("/").percent
    _d = bat = psutil.sensors_battery().percent
    data_queue.append([_a,_b,_c,_d])
    updateQuery(data_queue)
    print('...')
    
    time.sleep(0.1)
    #conn.commit()

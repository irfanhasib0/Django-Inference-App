from flask import Flask, render_template
from views import *

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('dash-board.html')

@app.route('/image_1',methods=['GET','POST'])
def call_1():
    return image_1()

@app.route('/image_2')
def call_2():
    return image_2()

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='7002',debug=True)
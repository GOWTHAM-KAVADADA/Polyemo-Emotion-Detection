

import os
from model import FacialExpressionModel
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from flask import Flask, redirect, url_for, request, render_template, Response
import librosa
import sqlite3
from keras.models import load_model
#import pandas as pd 
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)


UPLOAD_FOLDER1 = 'static/uploads/'


facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("models/model.json", "models/cnn.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER1


model_path2 = 'models/modelV2.h5' # load .h5 Model

model1 = load_model(model_path2, compile=False)

model_name = open("tk.pkl","rb")
scaler = pickle.load(model_name)


import warnings

lstm_model = load_model('lstm_model.h5')



cv=pickle.load(open('transform2.pkl','rb'))

def extract_features(data):
    result = np.array([])
    
    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    result = np.array(mfccs_processed)
    
    return result


model_path3 = 'models/image/cnn.h5' 

model_image = load_model(model_path3, compile=False)


@app.route("/index2")
def index2():
    return render_template("index2.html")
    
@app.route("/index")
def index():
    return render_template("index.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")

@app.route('/')
def home():
	return render_template('home.html')


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict2',methods=['GET','POST'])
def predict2():
    print("Entered")
    
    print("Entered here")
    file = request.files['files'] # fet input
    filename = file.filename        
    print("@@ Input posted = ", filename)
        
    file_path = os.path.join(UPLOAD_FOLDER1, filename)
    file.save(file_path)

    duration = 3 
    test_data, _ = librosa.load(file_path, duration=duration, res_type='kaiser_fast')
    test_features = extract_features(test_data)
    test_features = scaler.transform(test_features.reshape(1, -1))  
    test_features = np.expand_dims(test_features, axis=2)  

    
    predictions = model1.predict(test_features)
    predicted_class = np.argmax(predictions)
    print(predicted_class)

    return render_template('after.html', prediction = predicted_class)

    


@app.route('/predict',methods=['POST'])
def predict():

    text = request.form['message']
    #translations = translator.translate(text, dest='en')
    #message =  translations.text
    data = [text]
    vect = cv.texts_to_sequences(data)
    vect = pad_sequences(vect)
    k=np.zeros((1,64))
    k[0,-vect.shape[1]:]=vect
    my_prediction =  np.argmax(lstm_model.predict(k), axis=-1)
    predicted_class=my_prediction[0]

    return render_template('after.html', prediction = predicted_class)
    
    
@app.route('/predict1', methods=['POST'])
def model_predict1():
    if 'files' in request.files:
        image_file = request.files['files']
        if image_file.filename != '':
            
            image_path = 'temp_image.jpg'
            image_file.save(image_path)

            
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            image = image / 255
            image = np.expand_dims(image, axis=0)

            
            result = np.argmax(model_image.predict(image))
            print(result)

            return render_template('after.html', prediction = result)
    return "No file uploaded."



@app.route('/video')
def video():
	return render_template('video.html')

@app.route('/index1')
def index1():
	return render_template('index1.html')

if __name__ == '__main__':
    app.run(debug=False)
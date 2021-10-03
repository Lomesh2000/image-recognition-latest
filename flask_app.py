# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:53:20 2021

@author: lomes
"""

from flask import Flask,request,jsonify,render_template
import numpy as np
import keras
from PIL import Image
import flask
import os
from werkzeug.utils import secure_filename



app=Flask(__name__)
model=keras.models.load_model('image_recognition_model.h5')

classes = { 
    0:'aeroplane',
    1:'automobile',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck' 
}

def predict(file_path,model):
    im=Image.open(file_path)
    
    im=im.resize((32,32))
    im=np.expand_dims(im,axis=0)
    im=np.array(im)

    pred=model.predict(im)
    pred=np.argmax(pred)
    pred=classes[pred]
    pred=str(pred)
    return pred
    

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def upload():
    if request.method=='POST':
        
        file=request.files['file']
        
        filepath=os.path.join('D:/DEEP Learninf KN/practice/image recognition/uploads',secure_filename(file.filename))
        
        file.save(filepath)
        
        
        prediction=predict(filepath,model)
        print(prediction)
        
        return prediction
    
    return None

if __name__ == '__main__':
    app.run(debug=True)    
        
        
        
        
        
        
    
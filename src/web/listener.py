#!/usr/bin/env python
# encoding: utf-8
import json
from flask import Flask, jsonify, render_template, request, flash, redirect
from fl_client import *
import os
app = Flask(__name__)
app.secret_key = "abc"
@app.route('/train')
def train():
    name = 'client-1'
    setup_project(name)
    acc, loss = process()
    return jsonify({'Client': name, 'Accuracy': acc, 'Loss': loss})

@app.route('/api/test')
def test():
    from PIL import Image
    image = Image.open("/home/harsh_1921cs01/hub/os/nl/VJH_020/data/lemon/train_image_10/cat_0001/0001_F_V_75_C.jpg")
    health_index = health_meter(image)
    return jsonify({'Health Index': health_index})


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test1():
    return render_template('test.html')
                            
app.run(host='0.0.0.0', port=5000)
import os
import pandas as pd 
import json
import pickle
import time

from flask import Flask
from flask import render_template
from flask import jsonify , request
from flask import session , redirect

app = Flask(__name__,static_url_path='', 
            static_folder='static',
            template_folder='templates')


app.config['SECRET_KEY'] = 'secres'    



def get_preds(text: str):

    time.sleep(2)

    text =  '<p>I want a flight from <span class="badge bg-secondary">Mumbai</span> to <span class="badge bg-secondary">Delhi</span></p>'
    intent = 'Get Flight'
    entities = [('Source', 'Mumbai'), ('Destination', 'Delhi')]

    preds = {
        'intent': intent,
        'text': text,   
        'entities': entities
    }

    return preds


@app.route('/')
def index():
    return redirect('/home/')

@app.route('/home/')
def home():
    return render_template('home.html')

@app.route('/login/', methods=['GET'])
def login():
    return render_template('login.html')

@app.route('/datasets/', methods=['GET'])
def datasets():
    return render_template('datasets_view.html')

@app.route('/models/', methods=['GET'])
def model():
    return render_template('trained_models.html')

@app.route('/test/', methods=['GET', 'POST'])
def test():

    return render_template('test_model.html', preds=None)


@app.route('/prediction/', methods=["POST"])
def prediction():
    preds = get_preds("")

    return render_template('test_model.html', preds=preds)

@app.route('/train/', methods=['GET'])
def train():
    return render_template('train_model.html')
    
if __name__ == '__main__':

    app.run(debug=True)

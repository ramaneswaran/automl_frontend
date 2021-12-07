import os
import pandas as pd 
import json
import pickle
import time
from shutil import make_archive
import shutil
import datetime
import boto3
import botocore
from flask import Flask,flash
from flask import render_template
from flask import jsonify , request
from flask import session , redirect
import json
from flask_cors import CORS, cross_origin
from sklearn.preprocessing import LabelEncoder
import requests

app = Flask(__name__,static_url_path='', 
            static_folder='static',
            template_folder='templates')

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = 'secres'  
app.config['ENGINE_URL'] = 'http://localhost:8000'  

s3 = boto3.resource('s3',aws_access_key_id = os.environ.get("aws_access_key"), aws_secret_access_key = os.environ.get("aws_secret_key"),
         region_name = 'ap-south-1')


#Data parser class
class DataParser:

    def __init__(self, train_path, val_path):

        self.train_path = train_path
        self.val_path = val_path

        self.train_data = self._read_data(self.train_path)
        self.val_data = self._read_data(self.val_path)

        self.intent_encoder = self._train_intent_encoder()
        self.tag_encoder = self._train_tag_encoder()

    def _read_data(self, data_path):
        f = open(data_path, 'r')
        content = f.readlines()
        f.close()
        
        data = []
        
        for item in content:
            text, annotations = item.split('\t')
            
            sample = []
            
            for token, annotation in zip(text.split(), annotations.split()):
                sample.append((token, annotation))
        
            data.append(sample)
            
        return data

    def _train_intent_encoder(self):
        # Get all intents
        intents = set()
        annotations = set()
        for data in self.train_data:
            token, annotation = data[-1]
            intents.add(annotation)
                
        for data in self.val_data:
            token, annotation = data[-1]
            intents.add(annotation)

        intents = list(intents)

        # Fit label encoder for intents
        intent_encoder = LabelEncoder()
        intent_encoder.fit(intents)

        return intent_encoder

    def _train_tag_encoder(self):
        # Get all tags
        annotations = set()
        for data in self.train_data:
            for token, annotation in data[1:-1]:
                annotations.add(annotation)
                
        for data in self.val_data:
            for token, annotation in data[1:-1]:
                annotations.add(annotation)

        annotations = list(annotations)

        # Fit a label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(annotations)

        return label_encoder

    def write_config(self, save_dir):

        if os.path.exists(save_dir) is False:
            os.mkdir(save_dir)

        train_save_path = os.path.join(save_dir, 'train.pkl')
        with open(train_save_path, 'wb') as f:
            pickle.dump(self.train_data, f)

        val_save_path = os.path.join(save_dir, 'val.pkl')
        with open(val_save_path, 'wb') as f:
            pickle.dump(self.val_data, f)
        
        config = {
            'num_intents': len(self.intent_encoder.classes_),
            'num_tags': len(self.tag_encoder.classes_),
            'train_path': train_save_path,
            'val_path': val_save_path, 
        }

        # Serializing json 
        json_object = json.dumps(config, indent = 4)
        
        config_path = os.path.join(save_dir, 'config.json')

        # Writing to sample.json
        with open(config_path, "w") as outfile:
            outfile.write(json_object)
    

def get_datasets_names():

    dataset_bucket = s3.Bucket('automl-training-data-s3')
    dataset_names = [(idx+1, bucket_obj.key) for idx, bucket_obj in enumerate(dataset_bucket.objects.all())]

    # return [(1, "ATIS V1"), (2, "ATIS V2")]
    return dataset_names

def get_model_names():

    dataset_bucket = s3.Bucket('automl-models-s3')
    dataset_names = [(idx+1, bucket_obj.key) for idx, bucket_obj in enumerate(dataset_bucket.objects.all())]

    # return [(1, "Albertv2"), (2, "Roberta")]

    return dataset_names

def get_preds(text: str):

    url = f"{app.config['ENGINE_URL']}/predict/"
    result = requests.get(url, params={'text': text})

    # print(result.json())

    # text =  '<p>I want a flight from <span class="badge bg-secondary">Mumbai</span> to <span class="badge bg-secondary">Delhi</span></p>'
    # intent = 'Get Flight'
    # entities = [('Source', 'Mumbai'), ('Destination', 'Delhi')]

    # preds = {
    #     'intent': intent,
    #     'text': text,   
    #     'entities': entities
    # }

    return result.json()


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
    dataset_names = get_datasets_names()
    return render_template('datasets_view.html', dataset_names=dataset_names)

@app.route('/upload_datasets', methods=['GET'] )
def upload_datasets():
    return render_template('upload_datasets.html')

@app.route('/uploader', methods=['GET','POST'] )
def uploader():
    
    if request.method == 'POST':

        file_names = request.files.keys()

        dataset_name = request.form.get('name')

        current_time = str(datetime.datetime.now())

        temp_dir = 'temp_dir'

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        os.mkdir(temp_dir)

        data_paths = {}
        for file_name in file_names:
            save_path = os.path.join(temp_dir, request.files[file_name].filename)
            data_paths[file_name] = save_path
            request.files[file_name].save(save_path)

    
            

        parser = DataParser(data_paths['train'], data_paths['val'])
        parser.write_config(temp_dir)
    
        zip_file = f"{dataset_name}"  #zip file name 
        make_archive(zip_file, "zip", temp_dir)    
        
        try:
            s3.meta.client.upload_file(f"{zip_file}.zip", 'automl-training-data-s3', f"{zip_file}.zip")    
        except botocore.exceptions.ClientError as e:
            return {"status": e}  #response added 
        
        flash("Dataset uploaded successfully") 
        return redirect('/home/')

@app.route('/models/', methods=['GET'])
def model():
    model_names = get_model_names()
    return render_template('trained_models.html', model_names=model_names)

@app.route('/test/<model_name>', methods=['GET', 'POST'])
def test(model_name: str):
    return render_template('test_model.html', model_name=model_name)


@app.route('/prediction/', methods=["POST"])
def prediction():

    text = request.form['input_text']
    preds = get_preds(text)

    return render_template('test_model.html', preds=preds)

@app.route('/train/', methods=['GET', 'POST'])
def train():

    if request.method == 'GET': 
        dataset_names =  get_datasets_names()

        return render_template('train_model.html', dataset_names=dataset_names)
    
    elif request.method == 'POST':

        
        url = f"{app.config['ENGINE_URL'] }/train/"
        train_params = {
            'arch': request.form.get("arch"),
            'dataset': request.form.get('dataset')
            }

        x = requests.post(url, json=train_params)

        flash("Model Training In progress") 
        return render_template('home.html')


    
if __name__ == '__main__':

    app.run(debug=True)

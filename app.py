import os
import pandas as pd 
import json
import pickle
import time
from shutil import make_archive
import datetime
import boto3
import botocore
from flask import Flask,flash
from flask import render_template
from flask import jsonify , request
from flask import session , redirect
import json
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__,static_url_path='', 
            static_folder='static',
            template_folder='templates')


app.config['SECRET_KEY'] = 'secres'    

s3 = boto3.resource('s3',aws_access_key_id = os.environ.get("aws_access_key"), aws_secret_access_key = os.environ.get("aws_secret_key"),
         region_name = 'ap-south-1')


def get_datasets_names():
    pass 
    return [(1, "ATIS V1"), (2, "ATIS V2")]

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
    dataset_names = get_datasets_names()
    return render_template('datasets_view.html', dataset_names=dataset_names)

@app.route('/upload_datasets', methods=['GET'] )
def upload_datasets():
    return render_template('upload_datasets.html')

@app.route('/uploader', methods=['GET','POST'] )
def uploader():
    train_path = "../Dataset/atis-2.train.w-intent.iob"
    val_path = "../Dataset/atis-2.dev.w-intent.iob" 
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
    #                               *************************                          #
    
    if request.method == 'POST':
        file = request.files
        current_time = str(datetime.datetime.now())
    
        parser = DataParser(train_path, val_path)
        parser.write_config('parsed_data')
    
        zip_file = "data"  #zip file name 
        directory = "./parsed_data"
        make_archive(zip_file, "zip", directory)    
        try:
            s3.meta.client.upload_file('./data.zip', 'automl-training-data-s3', 'data'+current_time+'.zip')    
        except botocore.exceptions.ClientError as e:
            return {"status": e}  #response added 

        return 'success'

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

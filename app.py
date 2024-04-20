#  to run library python -m http.server
# env convo myapp
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import os
import pyaudio
import wave
from pydub import AudioSegment
from datetime import datetime
import numpy as np
from llama_index.core import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
import pickle
from knowledgebase import create_or_load_ko_dump,get_automerging_query_engine,MultiQueriesRetriever
app = Flask(__name__)
from faster_whisper import WhisperModel
    
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

CONTEXT = './knowledge_dump/WHO/service_context.pkl'

with open(CONTEXT,'rb') as f:
    merging_context=pickle.load(f)

automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f'./knowledge_dump/WHO'),
            service_context=merging_context,
        )
whisper_model = WhisperModel("Models/faster-whisper-small.en")
# whisper_model = WhisperModel("Models/faster-whisper-small")
# whisper_model = WhisperModel("Models/faster-distil-whisper-large-v2")
# whisper_model = WhisperModel("Models/faster-whisper-medium")
query_engine=get_automerging_query_engine(automerging_index,12,'refine')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploadfile', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'message': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Convert audio to text (replace this with your actual logic)
        audio = AudioSegment.from_wav(filepath)
        text = "Audio converted successfully"
        
        return jsonify({'message': text})
    
    return jsonify({'message': 'Invalid file format'})


@app.route('/input_text', methods=['POST'])
def input_text():
    input_text = request.form['text']
    return jsonify({'message': f'Received text input: {input_text}'})


@app.route('/transcribe', methods=['POST'])
def transcribe():
    global whisper_model
    audio_file = request.files['audio']
    segments, info = whisper_model.transcribe(audio_file)
    result=''
    for segment in segments:
        result+=segment.text
    print()
    return result


from flask import Response
import json
from flask_cors import CORS
CORS(app)  #
@app.route('/getanswer', methods=['POST', 'OPTIONS'])
def getanswer():
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '86400',
        }
        return '', 204, headers

    question = request.data.decode("utf-8")
    # question=MultiQueriesRetriever().gen_queries(question)
    response_gen = query_engine.query(question).response_gen

    def stream():
        # for chunk in response_gen:
        #     yield json.dumps(chunk).encode('utf-8')
        for chunk in response_gen:
            json_chunk = json.dumps(chunk)
            yield f"{json_chunk}\n" 

    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Access-Control-Allow-Origin': '*',
    }
    return Response(stream(), headers=headers)



@app.route('/books', methods=['GET'])
def get_book():
    folder_path = './Library'
    try:
        books = []
        for filename in os.listdir(folder_path):
            # Assuming the filename is the title and the full path is the path
            filename_t=filename
            if len(filename)>=35:
                filename_t=filename[0:35]+'...'
            if filename!='libquotes.png': books.append({'title': filename_t, 'path':  filename})
        return jsonify(books)
    except Exception as e:
        print(f'Error reading folder: {e}')
        return jsonify([]), 500


@app.route('/read_book', methods=['POST'])
def get_query_engine():
    try:
        file = request.files['file']
        filename = file.filename.replace(' ','_')
        print(f'reading {filename}................')
        index,storage_Context=create_or_load_ko_dump(filename)
        get_automerging_query_engine(index,storage_context=storage_Context ,similarity_top_k=20,response_mode='tree_summarize')
        return jsonify({'message': 'Thanks for your patience,I have completed reading your book'}),200
    except Exception as e:
        raise ValueError(f'Unable to read you File {e}',)
    


UPLOAD_FOLDER = 'Library'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/uploadpdf', methods=['POST'])
def add_to_Library():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = file.filename.replace(' ','_')
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully'}), 200
    




@app.route('/loadindex', methods=['POST'])
def loadindex():
    file = request.data.decode("utf-8")
    file=file.replace(' ','_')
    file=file.split('.')[0]
    CONTEXT = f'knowledge_dump/{file}/service_context.pkl'
    global query_engine
    global automerging_index
    with open(CONTEXT,'rb') as f:
        merging_context=pickle.load(f)

    automerging_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=f'knowledge_dump/{file}'),
                service_context=merging_context,
            )
    query_engine=get_automerging_query_engine(automerging_index,12,'refine')
    return jsonify({'message': 'you can ask me anything from your document'}),200







if __name__ == '__main__':
    app.run(debug=True)
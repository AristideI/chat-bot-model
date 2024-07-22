from flask import Flask,request,jsonify
from flask_cors import CORS
from chat import get_response

app= Flask(__name__)

CORS(app)

@app.post('/ask')
def index():
    # get the question from the front-end user
    question= request.get_json().get('question')

    # call the response function to predict the answer
    response= get_response(question)

    # return the result to the front-end user
    return jsonify({"answer":response})
    
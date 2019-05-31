import json
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from predictor import my_customer_predictor

app = Flask(__name__)
CORS(app)
@app.route("/customer/", methods=['GET'])
def return_cluster():

    age = request.args.get('age')
    gender = request.args.get('gender')
    income = request.args.get('income')
    spend = request.args.get('spend')

    p = my_customer_predictor()

    cluster = p.predict(age, gender, income, spend)

    cluster_dict = {
        'model': 'knn',
        'cluster': cluster.tolist()
    }

    # cluster_dict = { 'age': age, 'gender': gender, 'income': income, 'spend': spend }

    return jsonify(cluster_dict)


@app.route("/", methods=['GET'])
def default():
    return "<h1> Welcome to customer predictor <h1>"


if __name__ == "__main__":
    app.run()

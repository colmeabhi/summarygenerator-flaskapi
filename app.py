# coding: utf-8
from flask import Flask, jsonify, render_template, request

import model_features


app = Flask(__name__)


@app.route('/')
def index():
    """Serve a minimal UI for interacting with the summariser."""
    return render_template('index.html')


@app.route('/json-example', methods=['POST'])
def summary_creator():
    request_data = request.get_json()
    if not request_data or 'textString' not in request_data:
        return jsonify({'error': 'textString is required'}), 400

    text = request_data['textString']
    final_text = model_features.executeForAFile(text)

    return jsonify({'summary': final_text})


if __name__ == '__main__':
    app.run(debug=True, port=5000)


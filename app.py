# coding: utf-8
from flask import Flask, request
import model_features


# ********************************************************************************************************************

app = Flask(__name__)

@app.route('/json-example', methods=['POST'])
def Summary_Creator():
    request_data = request.get_json()
    text = None
    
    if request_data:
        if 'textString' in request_data:
            text = request_data['textString']
            finalText = model_features.executeForAFile(text)
     
    return "The given text is : {}".format(finalText)

if __name__ == '__main__':
    app.run(debug=True, port=5000)


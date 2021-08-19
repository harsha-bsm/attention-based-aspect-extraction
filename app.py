from flask import Flask
from flask import request
from flask import jsonify
from predict import *


app = Flask(__name__)

@app.route('/correct', methods=['GET'])
def aspect():
    try:
        text = request.args.get('text')
        results = topicpredict(text)
    except Exception as error:
        results = 'Unexpected error: {}'.format(error)
    out = {'input_sentence':text, 'correct_sent':results}
    out = jsonify(out)
    return out

if __name__ == '__main__':
    app.run()
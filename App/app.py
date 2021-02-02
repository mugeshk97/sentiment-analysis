from flask import Flask, request, render_template
from utils import sentiment
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def login():
    prob = ''
    if request.method == 'POST':
        tag = request.form['input']
        prob = sentiment(tag)
    return render_template('index.html', probability = prob)


if __name__ == "__main__":
    app.run(debug= True)
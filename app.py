from flask import Flask, render_template, request, jsonify
from model import detect_phishing

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    email_text = request.form['email_text']
    result = detect_phishing(email_text)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)

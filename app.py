from flask import Flask, render_template, request
from final import make_prediction

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        return make_prediction()
    else:
        return render_template('newindex.html')
    return render_template('newindex.html')


if __name__ == "__main__":
    app.run(debug=True)

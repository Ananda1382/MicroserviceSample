from flask import Flask, render_template, request
import socket
app = Flask(__name__)

@app.route('/test')
def hello_world():
    return 'Hello World! %s' % socket.gethostname()


if __name__ == '__main__':
    # app.run(debug=True, port=3000)
    app.run(host='0.0.0.0', debug=True, port=3000)
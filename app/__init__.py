from flask import Flask

app = Flask(__name__, static_url_path='')

if __name__ == '__main__':
    app.run('0.0.0.0')
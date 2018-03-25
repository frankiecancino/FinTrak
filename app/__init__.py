from flask import Flask, render_template, redirect

app = Flask(__name__, static_url_path='')


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/survey')
def survey():
    return redirect("https://umn.qualtrics.com/jfe/form/SV_82Gs5cK2ljX6egJ")


if __name__ == '__main__':
    app.run('0.0.0.0')
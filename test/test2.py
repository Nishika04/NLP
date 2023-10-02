from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def hello_world():
  return 'Hello, World!'

@app.route('/process', methods=['GET', 'POST'])
def my_function():
  if request.method == 'POST':
    input_string = request.form['input_string']
    print(input_string)
    processed_input_string = input_string.upper()

    return render_template('index.html', processed_input_string=processed_input_string)
  
  return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)
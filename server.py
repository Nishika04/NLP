from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from multiple_emotions_nlp_implementation_reddit import analysis

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:8080"}})
print("New script has been loaded")

# logging.getLogger('flask_cors').level = logging.DEBUG
# data= {"data":"What the hell"}



@app.route('/submit_data', methods=['POST'])
@cross_origin(origins=['*'])


def process():
        
        data = request.get_json().get('text')
        #data = request.args.get('text')
        
        result=analysis(data)

        
        response = {
            "result": result,
        }
        # response.headers.add('Access-Control-Allow-Origin','*')
        # response.headers.add('Access-Control-Allow-Methods','*')
        # response.headers.add('Access-Control-Allow-Headers','*')
        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from multiple_emotions_nlp_implementation_reddit import analysis
import pandas as pd
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:8080"}})
print("New script has been loaded")

@app.route('/submit_data', methods=['POST'])
@cross_origin(origins=['*'])
def process_excel():
    # Get the uploaded file from the request
    file = request.files['file']

    # Read the Excel file into a DataFrame
    df = pd.read_excel(file)

    # Apply the analysis function to a particular column ('text_column' in this example)
    df['Result'] = df['Text'].apply(analysis)

    # Save the modified DataFrame to an Excel file on disk
    processed_file_path = 'D:/Personal/Nishika Singhvi/Testing Output/processed_file.xlsx'
    df.to_excel(processed_file_path, index=False, sheet_name='Sheet1')

    return send_file(
        processed_file_path,
        as_attachment=True,
        download_name='processed_file.xlsx',
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


if __name__ == '__main__':
    app.run(debug=True)



# def process():
        
#         data = request.get_json().get('text')
       
        
#         result=analysis(data)

        
#         response = {
#             "result": result,
#         }

#         return jsonify(response)


# if __name__ == '__main__':
#     app.run(debug=True)

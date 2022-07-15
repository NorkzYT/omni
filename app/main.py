# import requirements needed
from flask import Flask, render_template,request
from utils import get_base_url
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
from sklearn.externals.joblib import dump, load
import numpy as np
import pickle

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12394
base_url = get_base_url(port)

def numbers_to_genre(argument):
    switcher = {0: 'Electronic', 1: 'Anime', 2: 'Jazz', 3: 'Alternative', 4: 'Country', 5: 'Rap', 6: 'Blues', 7: 'Rock', 8: 'Classical', 9: 'Hip-Hop'}
    return switcher.get(argument, "nothing")

sc_load = load('std_scaler.bin')
pickled_model = pickle.load(open('model.pkl', 'rb'))


# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

# set up the routes and logic for the webserver
@app.route(f'{base_url}')
def home():
    return render_template('website-code.html')



@app.route(f"{base_url}", methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return (flask.render_template('website-code.html', prediction_text = "",pred_val = 0))
    if request.method == 'POST':
        input_features = [float(item) for item in request.form.values()]
        input_array = np.array(input_features)
        scaled_input = sc_load.transform(input_array.reshape(1,-1))
        genre_prediction = numbers_to_genre(pickled_model.predict(scaled_input)[0])
        return render_template('website-code.html', prediction_text=genre_prediction,pred_val = 1)

    
    
# @app.route(f"{base_url}", methods=['GET', 'POST'])
# def main():
#     if request.method == 'GET':
#         return(flask.render_template('index.html', prediction_text = ""))
    
#     if request.method == 'POST':
        
#         inp_features = [float(x) for x in request.form.values()]
        
#         print(inp_features)
        
#         input_variables = np.append(inp_features,1)
        
#         input_variables = input_variables.reshape(1,-1)
       
#         prediction = model.predict(input_variables)[0]
        
#         return render_template('index.html',
#                                      prediction_text=prediction,
#                                      )

# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'https://cocalc13.ai-camp.dev'

    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)
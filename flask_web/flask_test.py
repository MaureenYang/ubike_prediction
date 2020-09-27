import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import json
import gzip

from flask import Flask, render_template, session, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField,DecimalField,DateField
from wtforms.validators import DataRequired
from wtforms.fields.html5 import DateTimeLocalField

from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map

df = pd.read_csv('src_csv/new_data_sno1_predict.csv')

# preprocessing
df = df.dropna()
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='ignore')
df = df.set_index(pd.DatetimeIndex(df['time']))
df = df.drop_duplicates()
X = df.drop(columns = [df.keys()[0],'sbi'])

model = pickle.load(open('lasso_new.pkl','rb'))

class InputForm(FlaskForm):
    station_id = DecimalField('Station ID', validators=[DataRequired()])
    current_time = DateTimeLocalField('Datetime',format='%Y-%m-%dT%H:%M')
    pred_hour = DecimalField('Predict Hour', validators=[DataRequired()])
    submit = SubmitField('Submit')

#create markers for all station
def station_markers():
    i = 0
    station_info = []
    ubike_f = gzip.open("src_csv/ubike_data.gz", 'r')
    ubike_jdata = ubike_f.read()
    ubike_f.close()
    ubike_data = json.loads(ubike_jdata)
    for key,value in ubike_data['retVal'].items():
        i = i + 1
        string_station = "["+value['sno']+"] " + value['sna'] + "<br> total: "+value['tot']+"<br>"
        val = ['bemp','sbi','act','mday','snaen','ar','aren','sarea','sareaen','sno', 'sna', 'tot']
        for e in val:
            value.pop(e)

        ubike_dict = value
        ubike_dict['infobox'] = string_station

        station_info = station_info + [ubike_dict]
        if i == 10:
            break
    return station_info

app = Flask(__name__)
app.config['GOOGLEMAPS_KEY'] = "AIzaSyBaZFXXOjxMDx7uqzfKRmVpbZl2XI51QPg"
app.config['SECRET_KEY'] = 'hard to guess string'
app.static_folder = 'static'
bootstrap = Bootstrap(app)
GoogleMaps(app)


@app.route('/',methods=['GET','POST'])
def index():
    prediction_text = ''
    inputform = InputForm()
    station_id = 1
    #markers for all station
    locations = station_markers()

    if inputform.validate_on_submit():
        station_id = int(inputform.station_id.data)
        current_time = inputform.current_time.data
        pred_hour = inputform.pred_hour.data

        ct = current_time.strftime("%Y%m%d %H:%M:%S")
        real_y = X[ct]
        real_y = real_y[real_y.predict_hour == pred_hour]
        real_y = real_y.drop_duplicates()                   #this is a problem
        X_pred = real_y.drop(columns=['bemp','time'])
        pred_y = int(model.predict(X_pred))
        prediction_text = 'In {}, the empty number of Station {} in {} hour after is {}, prediction is {}'.format(current_time,station_id,pred_hour,int(real_y.y_bemp.values),pred_y)
        new_str = "empty: "+ str(pred_y)+ "<br>"
        locations[station_id-1]['infobox'] = locations[station_id-1]['infobox'] + new_str

    bike_map = Map(
        identifier="tp-ubike-map",
        lat= locations[station_id-1]['lat'],
        lng= locations[station_id-1]['lng'],
        markers = locations,
        style= "height:550px; width:900px; margin:0;",
        #fit_markers_to_bounds = True
    )

    return render_template('index.html',form=inputform,prediction_text=prediction_text,bike_map=bike_map)


if __name__ == "__main__":
    app.run(debug=True)

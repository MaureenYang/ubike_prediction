import numpy as np
import pandas as pd
import pickle
from datetime import datetime

from flask import Flask, render_template, session, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField,DecimalField,DateField
from wtforms.validators import DataRequired
from wtforms.fields.html5 import DateTimeLocalField

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

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = 'hard to guess string'

@app.route('/',methods=['GET','POST'])
def index():
    prediction_text = ''
    inputform = InputForm()

    if inputform.validate_on_submit():
        station_id = inputform.station_id.data
        current_time = inputform.current_time.data
        pred_hour = inputform.pred_hour.data

        ct = current_time.strftime("%Y%m%d %H:%M:%S")
        real_y = X[ct]
        real_y = real_y[real_y.predict_hour == pred_hour]
        real_y = real_y.drop_duplicates()                   #this is a problem
        X_pred = real_y.drop(columns=['bemp','time'])
        pred_y = int(model.predict(X_pred))
        prediction_text = 'In {}, the empty number of Station {} in {} hour after is {}, prediction is {}'.format(current_time,station_id,pred_hour,int(real_y.y_bemp.values),pred_y)

    return render_template('index.html',form=inputform,prediction_text=prediction_text)



if __name__ == "__main__":
    app.run(debug=True)

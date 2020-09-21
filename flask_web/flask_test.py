from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.model_selection import PredefinedSplit
import pandas as pd

df = pd.read_csv('new_data_sno1_parsed2_predict_merged_6h.csv')

def index_splitter(N, fold):
    index_split = []
    test_num = int(N/fold)
    train_num = N-test_num

    for i in range(0,train_num):
        index_split.append(-1)

    for i in range(train_num,N):
        index_split.append(0)

    return index_split

# preprocessing
df = df.dropna()
X = df.drop(columns = [df.keys()[0],'tot','sbi','bemp','act'])
Y = df['bemp']

# Data Splitter
arr = index_splitter(N=len(X), fold=4)
ps = PredefinedSplit(arr)

for train, test in ps.split():
    train_index = train
    test_index = test


train_X, train_y = X.iloc[train_index,:], Y.iloc[train_index]
test_X, test_y = X.iloc[test_index,:], Y.iloc[test_index]
model = pickle.load(open('model.pkl', 'rb'))


app = Flask(__name__)
@app.route('/')
def home():
    return render_template('predict.html')
from datetime import datetime


@app.route('/predict',methods=['POST'])
def predict():
    int_features = []
    for x in request.form.values():
        print("x:", x)
        try:
            int_features = int_features + [int(x)]
        except:
            dt = datetime.strptime(x, "%Y/%m/%d %H:%M:%S")
            int_features = int_features + [dt]
            print("dt:", dt)

    #print("dt:", test_X.iloc(0))

    #final_features = [np.array(int_features)]
    #print(test_X[dt])
    prediction = model.predict(test_X)

    output = round(prediction[0])

    return render_template('predict.html', prediction_text='Sales should be $ {}'.format(output))
    #return render_template('predict.html', prediction_text='predict should be $ {}'.format(int_features))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)

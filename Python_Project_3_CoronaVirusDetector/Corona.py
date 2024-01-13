# We can use flask server for display UI

import pickle
from flask import Flask,render_template , request
app = Flask(__name__)


with open("Python_Project/Python_Project_3_CoronaVirusDetector/model.pkl","rb") as file:
    clf = pickle.load(file)
@app.route('/',methods=['GET','POST'])
def hello_world():
    if request.method == 'POST':
        dict = request.form
        fever = int(dict['fever'])
        age = int(dict['age'])
        pain = int(dict['pain'])
        runnyNose = int(dict['runnyNose'])
        diffBreath = int(dict['diffBreath'])
        # Input Value
        input = [fever, pain, age,runnyNose, diffBreath]
        infecProbability = clf.predict_proba([input])[0][1]
        print(infecProbability)
        return render_template('show.html',inf=round(infecProbability*100))
        # return 'Hello, World!' +str(infecProbability)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

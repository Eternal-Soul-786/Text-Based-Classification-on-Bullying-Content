from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import CORS
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import tensorflow as tf
import math
import bert
import tensorflow_hub as hub
import numpy

app = Flask(__name__)
app.config.from_object(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})



@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
	#df= pd.read_csv("YoutubeSpamMergedData.csv")
	#df_data = df[["CONTENT","CLASS"]]
	# Features and Labels
	#df_x = df_data['CONTENT']
	#df_y = df_data.CLASS
    # Extract Feature With CountVectorizer
	#corpus = df_x
	#cv = CountVectorizer()
	#X = cv.fit_transform(corpus) # Fit the Data
	#from sklearn.model_selection import train_test_split
	#X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
	#from sklearn.naive_bayes import MultinomialNB
	#clf = MultinomialNB()
	#clf.fit(X_train,y_train)
	#clf.score(X_test,y_test)
	
	#Alternative Usage of Saved Model
	# ytb_model = open("naivebayes_spam_model.pkl","rb")
	# clf = joblib.load(ytb_model)
	
	#ytb_model = open("naivebayes_spam_model.pkl","rb")
	#clf = joblib.load(ytb_model)
		
	new_model = tf.keras.models.load_model("kol")
	if request.method == 'POST':
		comment = request.form['comment']
        
		FullTokenizer = bert.bert_tokenization.FullTokenizer
		bert_layer = hub.KerasLayer("1", trainable=False)
		vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
		do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
		tokenizer = FullTokenizer(vocab_file, do_lower_case)	
		tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(comment))	
		inputs = tf.expand_dims(tokens, 0)
		output = new_model(inputs, training=False)  
		sentiment = math.floor(output*2)
		my_prediction = format(output)

        
	
	return render_template('result.html',prediction = sentiment)




if __name__ == '__main__':
	app.run(debug=True)

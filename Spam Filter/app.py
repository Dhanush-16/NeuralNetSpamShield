from flask import Flask,render_template,redirect,request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
model = load_model("spam_model.h5")
app=Flask(__name__)
import joblib
token=joblib.load("token.pkl")
@app.route("/")
def wellcome():
    return render_template("main.html")
@app.route("/analyze",methods=["POST","GET"])
def analyze():
    length=183
    result=None
    message=request.form.get("message")
    message=token.texts_to_sequences([message.lower()])
    l=pad_sequences(message, maxlen=length, padding="pre")
    prediction = model.predict(l)
    print(prediction)
    result="Spam" if prediction[0][0] > 0.5 else "Ham"
    return render_template("result.html",prediction=result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
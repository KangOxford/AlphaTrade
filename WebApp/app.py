from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

app = Flask(__name__)
model = VGG16()

@app.route("/", methods = ["GET"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(port = 3000, debug= True)
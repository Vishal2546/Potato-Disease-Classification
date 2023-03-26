from flask import Flask, render_template, request
from keras.utils import img_to_array, load_img
from keras.models import load_model
import os
import numpy as np


app = Flask(__name__)

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(THIS_FOLDER, 'model/1')
model = load_model(MODEL_PATH)

IMAGE_SIZE = 256
IMAGES_FOLDER = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

class_indices = {'early_blight': 0, 'healthy': 1, 'late_blight': 2}
class_names = ['early_blight', 'healthy', 'late_blight']

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    
    for f in os.listdir('static/images'):
        os.remove(os.path.join('static/images', f))

    imagefile = request.files['imagefile']

    if(not imagefile):
        return render_template('index.html', nofile="error")
    
    if(not allowed_file(imagefile.filename)):
        return render_template('index.html', notimage="error")

    
    full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
    imagefile.save(full_image_path)
    image = load_img(full_image_path, target_size=(IMAGE_SIZE,IMAGE_SIZE))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.

    pred = model.predict(image)
    predicted_class = class_names[np.argmax(pred)]


    return render_template('index.html', image=full_image_path, prediction=predicted_class)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=7000)

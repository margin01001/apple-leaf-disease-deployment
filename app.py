import os
from unicodedata import category
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np

# Because of some Warning I have to use below code
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
cnn_model = load_model('./Model/model.h5')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/images'
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

def model(file_dir):
    categories = ['Black Rot', 'Cedar Apple Rust', 'Apple Scab', 'Healthy']
    # loading image from dir and converting to np.array and reshaping for prediction
    test_image = image.img_to_array(image.load_img(file_dir, target_size=(150, 150))).reshape(-1, 150, 150, 3)
    pred = cnn_model.predict(test_image)
    print(pred)
    return categories[np.argmax(pred)]
    
@app.route('/predict', methods=['POST', "GET"])
def predict():
    file = request.files['file']
    file_dir = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(file_dir)
    result = model(file_dir)
    print(result)
    # removing image after predicted
    os.remove(file_dir)
    return render_template('predict.html', data = result)

if __name__ == '__main__':
    app.run(debug=True)

#secure_filename is used for:-
#input: secure_filename("../../../etc/passwd")
#output: 'etc_passwd'
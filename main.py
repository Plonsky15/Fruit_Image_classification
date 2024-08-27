
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'UTF-8'

# Define the working directory and paths
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'Fruit_image_class_CNN.h5')
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Define class names directly
classes = [
    'Apple 6', 'Apple Braeburn 1', 'Apple Crimson Snow 1', 'Apple Golden 1', 'Apple Golden 2',
    'Apple Golden 3', 'Apple Granny Smith 1', 'Apple hit 1', 'Apple Pink Lady 1', 'Apple Red 1',
    'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious 1', 'Apple Red Yellow 1', 'Apple Red Yellow 2',
    'Apricot 1', 'Avocado 1', 'Avocado ripe 1', 'Banana 1', 'Banana Lady Finger 1', 'Banana Red 1',
    'Beetroot 1', 'Blueberry 1', 'Cabbage white 1', 'Cactus fruit 1', 'Cantaloupe 1', 'Cantaloupe 2',
    'Carambula 1', 'Carrot 1', 'Cauliflower 1', 'Cherry 1', 'Cherry 2', 'Cherry Rainier 1',
    'Cherry Wax Black 1', 'Cherry Wax Red 1', 'Cherry Wax Yellow 1', 'Chestnut 1', 'Clementine 1',
    'Cocos 1', 'Corn 1', 'Corn Husk 1', 'Cucumber 1', 'Cucumber 3', 'Cucumber Ripe 1', 'Cucumber Ripe 2',
    'Dates 1', 'Eggplant 1', 'Eggplant long 1', 'Fig 1', 'Ginger Root 1', 'Granadilla 1', 'Grape Blue 1',
    'Grape Pink 1', 'Grape White 1', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink 1',
    'Grapefruit White 1', 'Guava 1', 'Hazelnut 1', 'Huckleberry 1', 'Kaki 1', 'Kiwi 1', 'Kohlrabi 1',
    'Kumquats 1', 'Lemon 1', 'Lemon Meyer 1', 'Limes 1', 'Lychee 1', 'Mandarine 1', 'Mango 1',
    'Mango Red 1', 'Mangostan 1', 'Maracuja 1', 'Melon Piel de Sapo 1', 'Mulberry 1', 'Nectarine 1',
    'Nectarine Flat 1', 'Nut Forest 1', 'Nut Pecan 1', 'Onion Red 1', 'Onion Red Peeled 1',
    'Onion White 1', 'Orange 1', 'Papaya 1', 'Passion Fruit 1', 'Peach 1', 'Peach 2', 'Peach Flat 1',
    'Pear 1', 'Pear 2', 'Pear 3', 'Pear Abate 1', 'Pear Forelle 1', 'Pear Kaiser 1', 'Pear Monster 1',
    'Pear Red 1', 'Pear Stone 1', 'Pear Williams 1', 'Pepino 1', 'Pepper Green 1', 'Pepper Orange 1',
    'Pepper Red 1', 'Pepper Yellow 1', 'Physalis 1', 'Physalis with Husk 1', 'Pineapple 1',
    'Pineapple Mini 1', 'Pitahaya Red 1', 'Plum 1', 'Plum 2', 'Plum 3', 'Pomegranate 1', 'Pomelo Sweetie 1',
    'Potato Red 1', 'Potato Red Washed 1', 'Potato Sweet 1', 'Potato White 1', 'Quince 1', 'Rambutan 1',
    'Raspberry 1', 'Redcurrant 1', 'Salak 1', 'Strawberry 1', 'Strawberry Wedge 1', 'Tamarillo 1',
    'Tangelo 1', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red 1',
    'Tomato Heart 1', 'Tomato Maroon 1', 'Tomato not Ripened 1', 'Tomato Yellow 1', 'Walnut 1',
    'Watermelon 1', 'Zucchini 1', 'Zucchini dark 1'
]

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image, target_size=(100, 100)):
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image, classes):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = classes[predicted_class_index]
    return predicted_class_name

# Streamlit App
# image upload 
image_path = os.path.join(base_dir, 'orange-fruit-logo-vector-illustration-template_395528-original.png')

st. image(image_path, width = 300)

st.title('Fruit Image Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model, uploaded_image, classes)
            st.success(f'Prediction: {prediction}')

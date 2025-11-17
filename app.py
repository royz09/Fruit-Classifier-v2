import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('fruit_classifier_model.keras')

# Load fruit information from JSON
with open('fruit_info.json', 'r') as f:
    fruit_info = json.load(f)

# Define class names (must match the order during training)
class_names = ['apple', 'banana', 'orange']

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((128, 128)) # Resize to the target input size of the model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Function to make predictions
def predict_fruit(img_array):
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(score)
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(score) * 100
    return predicted_class_name, confidence

# Streamlit UI
st.title('Fruit Image Classifier')
st.write('Upload an image of an apple, banana, or orange to classify it!')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    processed_img = preprocess_image(img)
    predicted_fruit, confidence = predict_fruit(processed_img)

    st.success(f"Prediction: {predicted_fruit.capitalize()} with {confidence:.2f}% confidence.")

    # Display fruit information
    if predicted_fruit in fruit_info:
        st.subheader(f"About {fruit_info[predicted_fruit]['name']}")
        st.write(f"Description: {fruit_info[predicted_fruit]['description']}")
        st.write(f"Benefits: {fruit_info[predicted_fruit]['benefits']}")
    else:
        st.info("No detailed information available for this fruit.")

print("app.py created successfully.")

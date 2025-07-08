import numpy as np
from PIL import Image
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
model = load_model("fine_tuned_mobilenet.keras")

# Define class names in order
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Classification function
def classify_image(img):
    try:
        # Validate input
        if img is None:
            return " Error: No image uploaded."

        # Resize to expected input shape
        img = img.resize((224, 224))
        img = img.convert("RGB")
        img_array = np.array(img, dtype=np.float32)

        # If image is grayscale or has 1 channel, convert to RGB
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            img_array = np.stack((img_array,) * 3, axis=-1)

        # Preprocess and expand dimensions
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = prediction[0][predicted_index]

        return f"🗑️ Predicted: {predicted_class} (Confidence: {confidence:.2%})"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create and launch Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Garbage Classification using MobileNetV2",
    description="Upload a garbage image to identify its category (cardboard, glass, metal, paper, plastic, or trash)."
)

iface.launch(pwa=True,share=True)
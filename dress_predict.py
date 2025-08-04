import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
fabric_prices = {
    "Banarasi": 7000,
    "Cambric": 1000,
    "Chanderi": 5000,
    "Chiffon":2500,
    "Chikankari": 2945,
    "Cotton":500,
    "Kanchipuram": 8000,
    "Khadi": 2655,
    "Mysore": 6000,
    "Poplin": 500,
    "Tussar": 5500,
    "Voile": 1299,
}
fabric_model = keras.models.load_model("fabric_model.h5")
def predict_fabric_and_price(image_path):
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = fabric_model.predict(image)
    print("Raw Prediction Output:", prediction) 
    fabric_labels = ["Banarasi", "Cambric", "Chanderi","Chiffon", "Chikankari","Cotton", "Kanchipuram", "Khadi", "Mysore", "Poplin", "Tussar", "Voile"]
    fabric_type = fabric_labels[np.argmax(prediction)]
    estimated_price = fabric_prices[fabric_type]
    return fabric_type, estimated_price
if __name__ == "__main__":
    fabric, price = predict_fabric_and_price("captured_dress.jpg")
    print(f"Detected Fabric: {fabric}")
    print(f"Estimated Price: ₹{price}")
    image = cv2.imread("captured_dress.jpg")
    cv2.putText(image, f"Fabric: {fabric}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Price: ₹{price}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Dress Analysis", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

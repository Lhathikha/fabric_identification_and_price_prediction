import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32  
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)
train_data = train_datagen.flow_from_directory(
    "fabric_dataset/train",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
val_data = val_datagen.flow_from_directory(
    "fabric_dataset/validation",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
if val_data.samples == 0:
    raise ValueError("No images found in the validation dataset. Add more images.")
fabric_model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation="relu"),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(12, activation="softmax")  # 12 classes now
])
fabric_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
fabric_model.fit(train_data, validation_data=val_data, epochs=10)
fabric_model.save("fabric_model.h5")
print("Model saved as fabric_model.h5")

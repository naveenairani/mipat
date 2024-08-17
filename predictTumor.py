import cv2 as cv
import imutils
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model

# Load and fine-tune a pre-trained model
def load_finetune_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(240, 240, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Predict tumor
def predict_tumor(model, image):
    try:
        # Convert image to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        # Threshold the image
        thresh = cv.threshold(gray, 45, 255, cv.THRESH_BINARY)[1]
        thresh = cv.erode(thresh, None, iterations=2)
        thresh = cv.dilate(thresh, None, iterations=2)

        # Find contours
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)

        # Find extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        # Crop the new image
        new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

        # Resize and normalize the image
        image = cv.resize(new_image, dsize=(240, 240), interpolation=cv.INTER_CUBIC)
        image = image / 255.0

        # Reshape the image
        image = image.reshape((1, 240, 240, 3))

        # Make prediction
        res = model.predict(image)

        return res
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Load model
print("Loading and fine-tuning model...")
model = load_finetune_model()
print("Model loaded and fine-tuned successfully.")

# Test the function with a sample image
if __name__ == "__main__":
    # Update the path to your sample image
    test_image_path = 'path_to_your_sample_image.jpg'  # Replace with the correct path
    image = cv.imread(test_image_path)
    if image is None:
        print("Error: Image not found or unable to load.")
    else:
        print("Image loaded successfully.")

        # Predict tumor
        result = predict_tumor(model, image)
        if result is not None:
            print("Prediction result:", result)
        else:
            print("Prediction failed.")

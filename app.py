import scipy

from flask import Flask, render_template, request
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16

app = Flask(__name__)
model = VGG16()

# Create an ImageDataGenerator with data augmentation settings
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

@app.route("/", methods=["GET"])
def hello_world():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def predict():
    imagefile = request.files["imagefile"]
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # Apply data augmentation on the image
    augmented_images = datagen.flow(image, batch_size=1)

    # Get the prediction from the augmented image
    yhat = model.predict(augmented_images)
    label = decode_predictions(yhat)
    label = label[0][0]

    classification = "%s (%.2f%%)" % (label[1], label[2] * 100)

    return render_template("index.html", prediction=classification)

if __name__ == "__main__":
    app.run(port=3000, debug=True)

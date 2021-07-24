from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from keras.models import load_model
import io
import numpy as np

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

dic = {0 : 'Canh Xuan', 1 : 'Ronaldo', 2: 'Tuan Kha'}

   
model = load_model('model_fl.h5')
model.make_predict_function()


def get_img(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


@app.route("/predict", methods=["POST", "GET"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = get_img(image, target=(100, 100))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict_classes(image)

            data["predictions"] = [dic[preds[0]]]

            # loop over the results and add them to the list of
            # returned predictions

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
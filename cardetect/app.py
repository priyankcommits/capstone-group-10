# sys
import os, time, base64
from threading import Thread
from pathlib import Path

# web
from flask import Flask, render_template, request, jsonify, make_response
from werkzeug import secure_filename

# local
from model import start_data_prep, display_image, build_model, prepare_dataset, predict_car
from utils import put_s3_object, get_s3_object, get_s3_list, mandatory_data

app = Flask(__name__)

uploads_dir = os.path.join(app.instance_path, 'uploads')
models_dir = os.path.join(app.instance_path, 'models')
os.makedirs(uploads_dir, exist_ok=True)


@app.route('/', methods=["GET"])
def index():
    contents = get_s3_list("capstonegroup10", "data")
    sets = []
    for content in contents:
        key = content["Key"].split("/")[1]
        if key and key not in sets:
            sets.append(key)
    models = get_s3_list("capstonegroup10", "models")
    model_names = []
    for model in models:
        key = model["Key"].split("/")[1]
        if key:
            key = model["Key"].replace(".h5", "").split("_")[1]
            if key and key not in model_names:
                model_names.append(key)
    return render_template('index.html', data={"sets": sets, "models": model_names})

@app.route('/prepare', methods=["POST"])
def prepare_data():
    mandatory_data()
    file_annot_train = request.files.getlist("0")[0]
    file_annot_test = request.files.getlist("0")[1]
    base_output_dir = os.path.join(uploads_dir)
    path_train = os.path.join(uploads_dir, secure_filename(file_annot_train.filename))
    path_test = os.path.join(uploads_dir, secure_filename(file_annot_test.filename))
    file_annot_train.save(path_train)
    file_annot_test.save(path_test)
    data = start_data_prep(path_train, path_test)
    image_outputs = []
    for path,loc,im_class in data[0].shuffle(buffer_size=40).take(3):
        image_outputs.append(display_image(path_train, path,loc, im_class, 1))
    put_s3_object("capstonegroup10", path_train, "data/{}/annot_train.csv".format(request.values["name"]))
    put_s3_object("capstonegroup10", path_test, "data/{}/annot_test.csv".format(request.values["name"]))
    return make_response(jsonify(image_outputs), 200)

@app.route('/train', methods=["POST"])
def train_model():
    if os.environ["FLASK_ENV"] != "development":
        return "Training model in non local instances is disabled due to insufficient cloud resources"
    mandatory_data()
    dataset = request.values["name"]
    train_path = "data/{}/annot_train.csv".format(dataset)
    path_train = os.path.join(uploads_dir, secure_filename("annot-train-downloaded.csv"))
    test_path = "data/{}/annot_test.csv".format(dataset)
    path_test = os.path.join(uploads_dir, secure_filename("annot-test-downloaded.csv"))
    file_annot_train = get_s3_object("capstonegroup10", train_path, path_train)
    file_annot_test = get_s3_object("capstonegroup10", test_path, path_test)
    def thread_callback(train, test, model_path):
        build_model(train, test, model_path)
        put_s3_object("capstonegroup10", model_path, "models/Model_{}.h5".format(dataset))
    data = start_data_prep(path_train, path_test)
    thread = Thread(target=thread_callback, args=[data[2], data[3], models_dir + "/Model-{}.h5".format(dataset)])
    thread.start()
    return "Training has started this will take a lot of time, come back later use this model in the 'Detector' tab"

@app.route('/predict', methods=["POST"])
def predict():
    mandatory_data()
    dataset = request.values["name"]
    train_path = "data/{}/annot_train.csv".format(dataset)
    path_train = os.path.join(uploads_dir, secure_filename("annot-train-downloaded.csv"))
    test_path = "data/{}/annot_test.csv".format(dataset)
    path_test = os.path.join(uploads_dir, secure_filename("annot-test-downloaded.csv"))
    file_annot_train = get_s3_object("capstonegroup10", train_path, path_train)
    file_annot_test = get_s3_object("capstonegroup10", test_path, path_test)
    data = start_data_prep(path_train, path_test)
    image_uploaded = request.files.getlist("0")[0]
    image_path = os.path.join(uploads_dir, secure_filename("temp.jpg"))
    image_uploaded.save(image_path)
    model_path = "Model_{}.h5".format(dataset)
    full_path = models_dir + "/" + model_path
    cloud_path = "models/Model_{}.h5".format(dataset)
    required_file = Path(full_path)
    if not required_file.is_file():
        get_s3_object("capstonegroup10", cloud_path, full_path)
    else:
        print("exists locally, not downloading")
    result = predict_car(image_path, models_dir + "/" + model_path, data[4])
    return make_response(jsonify({"top": result[0], "image": result[1]}), 200)

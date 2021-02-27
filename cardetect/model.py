import os, base64
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


paths = {}
New_Height = 363
New_Width = 525
New_Size = [New_Height, New_Width]


def prepare_csv(path, images_path):
    dataframe = pd.read_csv(path, names=["Image Name", "X", "Y", "DX", "DY", "Image Class"])
    dataframe = dataframe.set_index("Image Name", drop=False)
    dataframe["Class"] = pd.Series(dtype="float64")
    dataframe["Address"] = pd.Series(dtype="string")
    dataframe["Image Height"] = pd.Series(dtype="float64")
    dataframe["Image Width"] = pd.Series(dtype="float64")
    folders = os.listdir(images_path)
    for i in folders:
      file_list = os.listdir(images_path+i)
      for name in file_list:
          dataframe["Class"][name] = i
          dataframe["Address"][name] = images_path + i + "/" + name
    return dataframe

### Raw Data set preparation function - Returns raw dataset
def prepare_dataset(type):
    import tensorflow as tf
    train_path_images = "car_data/car_data/train/"
    test_path_images = "car_data/car_data/test/"
    Class_Names = {}
    Annotation_Dict = {}
    if type == "train":
        annot_train = prepare_csv(paths["train_path"], train_path_images)
        dataset_raw = tf.data.Dataset.from_tensor_slices(
            (annot_train["Address"], annot_train[["X", "Y", "DX", "DY"]], annot_train["Image Class"]))
        Class_Names = annot_train["Class"].unique()
        Annotation_Dict = {annot_train[annot_train["Class"] == i]["Image Class"].unique()[
            0]: i for i in Class_Names}
    elif type == "test":
        annot_test = prepare_csv(paths["test_path"], test_path_images)
        dataset_raw = tf.data.Dataset.from_tensor_slices(
            (annot_test["Address"], annot_test[["X", "Y", "DX", "DY"]], annot_test["Image Class"]))
    return (dataset_raw, Annotation_Dict)


def preprocess_train_image(path, loc, im_class):
    import tensorflow as tf
    image = tf.io.read_file(path)
    Depth = len(prepare_dataset("train")[1])
    image = tf.image.decode_jpeg(image, channels=3)
    Height = tf.shape(image)[0]
    Width = tf.shape(image)[1]
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize(image, New_Size)
    #image = image/255.0

    ## Defining new location Coordinates
    New_X = tf.cast(loc[0], dtype=tf.float32)/tf.cast(Width, dtype=tf.float32) * \
        tf.cast(New_Width, dtype=tf.float32)  # Normalise bounding box by image size
    New_Y = tf.cast(loc[1], dtype=tf.float32)/tf.cast(Height, dtype=tf.float32) * \
        tf.cast(New_Height, dtype=tf.float32)  # Normalise bounding box by image size

    New_DX = tf.cast(loc[2], dtype=tf.float32)/tf.cast(Width, dtype=tf.float32) * \
        tf.cast(New_Width, dtype=tf.float32)  # Normalise bounding box by image size
    New_DY = tf.cast(loc[3], dtype=tf.float32)/tf.cast(Height, dtype=tf.float32) * \
        tf.cast(New_Height, dtype=tf.float32)  # Normalise bounding box by image size

    loc = tf.stack([New_X, New_Y, New_DX-New_X, New_DY-New_Y])
    im_class = tf.one_hot(im_class-1, depth=Depth)
    target = {"Loc": loc, "Label": im_class}

    return (image, target)


def gen_main_data(data, batch_size):
    import tensorflow as tf
    data = data.shuffle(buffer_size = 10000)
    data = data.prefetch (buffer_size = tf.data.AUTOTUNE)
    data = data.map(preprocess_train_image,num_parallel_calls = tf.data.AUTOTUNE)
    data = data.batch(batch_size=batch_size)
    data = data.prefetch (buffer_size = tf.data.AUTOTUNE)
    return data

def start_data_prep(train_path, test_path):
    paths["train_path"] = train_path
    paths["test_path"] = test_path
    (train_dataset_raw, Annotation_Dict) = prepare_dataset("train")
    test_dataset_raw = prepare_dataset("test")[0]
    train_data = gen_main_data(train_dataset_raw, 32)
    test_data = gen_main_data(test_dataset_raw, 32)
    return (train_dataset_raw, test_dataset_raw, train_data, test_data, Annotation_Dict)

def display_image (train_path, path,loc,im_class,resized): ## 0 for no scaling, 1 for scaling
    import tensorflow as tf
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image,channels = 3)
    x1 = loc[0]
    x2 = loc[2]
    y1 = loc[1]
    y2 = loc[3]
    if resized == 1:
        x1 = (loc[0]/image.shape[1])*New_Width
        x2 = (loc[2]/image.shape[1])*New_Width
        y1 = (loc[1]/image.shape[0])*New_Height
        y2 = (loc[3]/image.shape[0])*New_Height
        image = tf.image.resize(image,New_Size)
        image = np.round(image.numpy(),0).astype(int)

    fig,ax=plt.subplots(1)
    ax.imshow(image)
    ax.add_patch(matplotlib.patches.Rectangle((x1,y1),width=x2-x1,height=y2-y1,facecolor="None",linewidth="2",edgecolor="r"))
    fig.savefig('temp.png')
    plot_file = open('temp.png', 'rb+')
    base64_string = base64.b64encode(plot_file.read()).decode()
    plot_file.close()
    return base64_string

def IOU(y_true, y_pred):
    import tensorflow as tf
    intersections = 0
    unions = 0
    # set the types so we are sure what type we are using

    gt = y_true
    pred = y_pred
    # Compute interection of predicted (pred) and ground truth (gt) bounding boxes
    diff_width = np.minimum(gt[:,0] + gt[:,2], pred[:,0] + pred[:,2]) - np.maximum(gt[:,0], pred[:,0])
    diff_height = np.minimum(gt[:,1] + gt[:,3], pred[:,1] + pred[:,3]) - np.maximum(gt[:,1], pred[:,1])
    intersection = diff_width * diff_height

    # Compute union
    area_gt = gt[:,2] * gt[:,3]
    area_pred = pred[:,2] * pred[:,3]
    union = area_gt + area_pred - intersection

    # Compute intersection and union over multiple boxes
    for j, _ in enumerate(union):
      if union[j] > 0 and intersection[j] > 0 and union[j] >= intersection[j]:
        intersections += intersection[j]
        unions += union[j]

    # Compute IOU. Use epsilon to prevent division by zero
    iou = np.round(intersections / (unions + tf.keras.backend.epsilon()), 4)
    # This must match the type used in py_func
    iou = iou.astype(np.float32)
    return iou

def IoU(y_true, y_pred):
    import tensorflow as tf
    iou = tf.py_function(IOU, [y_true, y_pred], Tout=tf.float32)
    return iou

def build_model(train_data, test_data, path_to_save):
    import tensorflow as tf
    input_shape = [New_Height,New_Width,3]
    InceptionResNetV2_layer = tf.keras.applications.InceptionResNetV2(input_shape=input_shape,include_top=False,weights="imagenet")
    for layer in InceptionResNetV2_layer.layers:
        layer.trainable= False

    input=tf.keras.Input(input_shape)
    preprocess = tf.keras.applications.inception_resnet_v2.preprocess_input(input)

    x0 = InceptionResNetV2_layer((preprocess))

    x0_0 = (x0)
    x1_2 = tf.keras.layers.Flatten()(x0_0)#(x1_1)
    x1_4 = tf.keras.layers.Dense(512,activation="swish")(x1_2)
    x1_5 = tf.keras.layers.BatchNormalization()(x1_4)
    x1_6 = tf.keras.layers.Dropout(0.3)(x1_5)
    x2_4 = tf.keras.layers.Dense(256,activation="swish")(x1_6)
    x2_5 = tf.keras.layers.BatchNormalization()(x2_4)
    x2_6 = tf.keras.layers.Dropout(0.3)(x2_5)
    loc = tf.keras.layers.Dense(4,name="Loc",activation="swish")(x2_6)

    y1_0 = tf.keras.layers.Flatten()(x0)
    y3_1 = tf.keras.layers.Dropout(0.80)(y1_0)
    y3_0 = tf.keras.layers.Dense(196,activation="softmax",name="Label",kernel_regularizer=tf.keras.regularizers.l1(0.000001))(y3_1)
    output = [loc,y3_0]

    model = tf.keras.Model(inputs = input,outputs = output )
    Saving_Callbacks_1 = tf.keras.callbacks.ModelCheckpoint(path_to_save + "softmax_model_top_k_label 2_20210214.h5",save_best_only=True,verbose=1,mode="max",monitor="val_Label_top_k_categorical_accuracy")
    Saving_Callbacks_2 = tf.keras.callbacks.ModelCheckpoint(path_to_save + "softmax_model_label_2_20210214.h5",save_best_only=True,verbose=1,mode="max",monitor="val_Label_categorical_accuracy")

    Reduce_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_Label_categorical_accuracy", factor=0.85, patience=2,verbose=1,mode = "max",min_delta=0.008)
    top_k_categorical_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_k_categorical_accuracy', dtype=None)

    #adam=tf.keras.optimizers.Adam(learning_rate=0.00008,beta_1=0.9,beta_2=0.999, epsilon=1e-07,amsgrad=False,name="Adam")
    Nadam = tf.keras.optimizers.Nadam(learning_rate=0.1, beta_1=0.99, beta_2=0.999, epsilon=1e-07, name="Nadam")

    model.compile(
        loss={"Loc":"mean_squared_error","Label":tf.keras.losses.CategoricalCrossentropy (from_logits=False)},
        loss_weights=[1,1],
        optimizer=Nadam,
        metrics={"Loc":IoU,"Label": [top_k_categorical_accuracy, "categorical_accuracy"]}
    )

    history = model.fit(train_data,epochs=100,callbacks=[Reduce_LR,Saving_Callbacks_1,Saving_Callbacks_2],validation_data=test_data)
    tf.keras.models.save_model(filepath=path_to_save, model=model)
    return


def predict_car(image_path, path_to_load, Annotation_Dict):
    import tensorflow as tf
    loaded_model = tf.keras.models.load_model(path_to_load, custom_objects={"IoU": IoU})
    predictions = show_picture( Annotation_Dict, image_path, loaded_model)
    return predictions

def read_image(path):
    import tensorflow as tf
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image,channels = 3)
    image = tf.cast(image,dtype = tf.float32)
    image = tf.image.resize(image,New_Size)
    return image

def create_test_data(path):
    import tensorflow as tf
    data_image = tf.data.Dataset.from_tensors(path)
    data_image = data_image.map(read_image)
    data_image = data_image.batch(batch_size=1)
    return data_image

### Call show_picture(test image path) for prediction
def show_picture(Annotation_Dict, path, model):
    main_image = plt.imread(path)
    old_x = main_image.shape[1]
    old_y = main_image.shape[0]

    eval_image = create_test_data(path)
    (image_coordinates,image_class_prob) = model.predict(eval_image)

    image_coordinates=image_coordinates[0]
    image_class = np.argmax(image_class_prob[0])+1 ## reduced 1 during trainnning
    image_class = Annotation_Dict[image_class]
    x1 = image_coordinates[0]
    y1 = image_coordinates[1]
    dx = image_coordinates[2]
    dy = image_coordinates[3]

    x1 = x1*old_x/New_Width
    y1 = y1*old_y/New_Height
    dx = dx*old_x/New_Width
    dy = dy*old_y/New_Height


    fig, ax1 = plt.subplots(1)
    ax1.imshow(main_image)
    ax1.add_patch(matplotlib.patches.Rectangle((x1,y1),width=dx,height=dy,facecolor="None",linewidth="2",edgecolor="r"))
    fig.savefig('temp.png')
    plot_file = open('temp.png', 'rb+')
    base64_string = base64.b64encode(plot_file.read()).decode()
    plot_file.close()

    top_5 = np.argsort(image_class_prob)[:,-5:]

    top_5_class=[]
    top_5_prob=[]
    for i in top_5[0]: #range(len(top_5)):
      top_5_prob.append(image_class_prob[0][i])
      top_5_class.append(Annotation_Dict[i+1])
    return (top_5_class, base64_string, image_class)

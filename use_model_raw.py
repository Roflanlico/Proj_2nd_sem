import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from io import BytesIO
from collections import Counter
from tensorflow import keras
from keras.models import load_model

def cv2_to_pil(img_cv2):
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    image_pillow = Image.fromarray(img_cv2)
    return image_pillow


def pil_to_cv2(img_pil):
    tmp = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return tmp

def load_image():
    img = Image.open("test_image.jpg")
    return img

def load_model_2():
    # Загрузка модели TensorFlow с Google Диска
    model = tf.saved_model.load('saved_model_6')
    return model

def crop_img(img):
    tmp = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    height, width, chans = tmp.shape
    image = tmp
    # Получение размеров изображения
    height, width = image.shape[:2]

    # Определение размера центральной части
    center_x, center_y = height // 2,  width// 2

    crop_size_x = (height // 224 - 0) * 224
    crop_size_y = (width // 224 - 0) * 224
    crop_size_x = 224 if crop_size_x <= 0 else crop_size_x
    crop_size_y = 224 if crop_size_y <= 0 else crop_size_y

    # Определение координат для обрезки
    x1, y1 = center_x - crop_size_x // 2, center_y - crop_size_y // 2
    x2, y2 = center_x + crop_size_x // 2, center_y + crop_size_y // 2

    # Вырезание центральной части изображения
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def slice_image(image_PIL, chunk_size):
    image = image_PIL
    width, height = image.size
    tmp_set = []
    for x in range(0, width, chunk_size):
        for y in range(0, height, chunk_size):
            box = (x, y, x + chunk_size, y + chunk_size)
            region = image.crop(box)
            tmp_set.append(region)
    return tmp_set

chunk_size = 224
tmp = []
classes = ['Amarant', 'Cabbage', 'Watercress']

def read_and_preprocess_image(image):
    img = image
    # img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

def use_model(model, tensor):
    #infer = model.signatures['serving_default']
    input_data = tensor
    output = model(input_data)
    return output

def predict(img):
    model = load_model_2()
    images = slice_image(cv2_to_pil(crop_img(img)), 224)
    predictions = []
    for img1 in images:
        image = read_and_preprocess_image(img1)
        pred = use_model(model, image)
        #print(pred)
        predictions.append(np.argmax(pred))
    #print(predictions)
    counter = Counter(predictions)
    #print(counter)
    #print(counter.most_common(1))
    #print(counter.most_common(1)[0])
    #print(counter.most_common(1)[0][0])
    most_common_element = counter.most_common(1)[0][0]
    predictions.clear()
    #print(predictions)
    return most_common_element

img = load_image()
print(classes[(predict(img))])

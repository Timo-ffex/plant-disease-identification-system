import cv2
import os
import numpy as np
from keras.preprocessing import image



from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# camera = cv2.VideoCapture(0)




def model_predict(img_path, model, plant, delete):
    

    IMAGE_SIZE = [224, 224]

    if plant == 'tomato':
        labels = ['Bacterial_spot','Early_blight','Late_blight','Leaf_Mold','Septoria_leaf_spot',
                'Spider_mites','Target_Spot','Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato_mosaic_virus','healthy']
    
    if plant == 'potato':
        labels = ['Early_blight', 'Late_blight', 'healthy']

    if plant == 'pepper':
        labels = ['Bacterial_spot', 'Healthy']

    if plant == 'maize':
        labels = ['Cercospora_leaf_spot', 'Common_rust', 'Northern_Leaf_Blight', 'healthy']


    if plant == 'cassava':
        labels =  ['cbb', 'cbsd', 'cgm', 'cmd', 'healthy']

    print('\n\n', img_path, '\n')
    img = image.load_img(img_path, target_size=IMAGE_SIZE)

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.array(x).astype('float32')/255
    x = np.reshape(x, IMAGE_SIZE + [3])
    x = np.expand_dims(x, axis=0)

    result = model.predict(x)
    print('\n\nDelete:\t', delete, '\n\n')
    if delete:
        os.remove(img_path)
        print('deleted ', img_path)

    what_class = np.argmax(result, axis=-1)
    scale = '{:.2f}'.format(round(result.max(), 2))

    result_label = labels[what_class[0]]
    label = result_label.replace('_', ' ').upper()
    if result_label.lower() == 'healthy':
        return (f'{plant.title()} leaf is classified HEALTHY with scale of {scale}'), 'HEALTHY', scale

    else:
        return (f'{plant.title()} leaf is infected with {label} with confident scale of {scale}'), label, scale


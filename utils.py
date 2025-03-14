from PIL import Image
from PIL import UnidentifiedImageError
import datetime
import time
import numpy as np 

def transform_image(object, type, train_transforms, val_transforms, label_map):

    img = None

    try:
        img = Image.open(object)

    except UnidentifiedImageError:

        print("PIL UnidentifiedImageError Bug caught!")

        dummy_image_data = np.random.rand(224, 224, 3)
        dummy_image_array = (dummy_image_data * 255).astype(np.uint8)
        img = Image.fromarray(dummy_image_array)


    if img.mode != "RGB":
        img = img.convert("RGB")        

    transforms_to_apply = None
    if type == "train":
        transforms_to_apply = train_transforms
    elif type == "val":
        transforms_to_apply = val_transforms

    transformed_img = transforms_to_apply(img)

    img.close()

    
    class_name = object.key.split("/")[1]
    class_idx = label_map[class_name]

    return (transformed_img, class_idx)

    
       

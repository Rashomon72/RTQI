import os
from ultralytics import YOLO
from collections import Counter
from statistics import mode
import numpy as np

def logicCases(results):

    resAll = []

    for result in results:
        resAll.append(result[0].boxes.cls.numpy().astype(int))

    count_All = np.zeros((len(resAll), 5), dtype=int)

    for i, res in enumerate(resAll):
        for ind in res:
            count_All[i][ind] += 1

    store = []

    for count in count_All:
        if count[1] > 0 and count[2] > 0 and count[4] > 0:
            store.append(3)
        elif count[1] > 0 and count[4] > 0:
            store.append(2)
        else:
            store.append(1)

    return mode(store)

def Lane_Markings(data_root, LM_model, images_dir):
    image_dir = os.path.join(data_root, images_dir).replace("\\", "/")

    results = []

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)

            results.append(LM_model(image_path))

    return logicCases(results)

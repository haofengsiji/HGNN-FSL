import os,glob
from PIL import Image
import pickle
import numpy as np
from tqdm import tqdm

CUB_200_2011_ROOT = "cub"

class_paths = os.listdir(CUB_200_2011_ROOT)

data = {}
for i in tqdm(range(200)):
    if i%2 != 0:
        continue
    image_paths = glob.glob(os.path.join(CUB_200_2011_ROOT,class_paths[i],'*.jpg'))
    for file in image_paths:
        img = Image.open(file).convert('RGB')
        img = np.asarray(img, dtype="uint8")
        if i in data:
            data[i].append(img)
        else:
            data[i] = []
            data[i].append(img)
pickle.dump(data,open('cub200_train.pickle','wb'),protocol=2)
print('cub200_trian.pickle done!')

data = {}
for i in tqdm(range(200)):
    if i%4 != 1:
        continue
    image_paths = glob.glob(os.path.join(CUB_200_2011_ROOT,class_paths[i],'*.jpg'))
    for file in image_paths:
        img = Image.open(file).convert('RGB')
        img = np.asarray(img, dtype="uint8")
        if i in data:
            data[i].append(img)
        else:
            data[i] = []
            data[i].append(img)
pickle.dump(data,open('cub200_val.pickle','wb'),protocol=2)
print('cub200_val.pickle done!')
data = {}
for i in tqdm(range(200)):
    if i%4 != 3:
        continue
    image_paths = glob.glob(os.path.join(CUB_200_2011_ROOT,class_paths[i],'*.jpg'))
    for file in image_paths:
        img = Image.open(file).convert('RGB')
        img = np.asarray(img, dtype="uint8")
        if i in data:
            data[i].append(img)
        else:
            data[i] = []
            data[i].append(img)
pickle.dump(data,open('cub200_test.pickle','wb'),protocol=2)
print('cub200_test.pickle done!')
    
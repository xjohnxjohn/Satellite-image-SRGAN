import os
import cv2
import numpy as np

INPUT ='./source/'

OUTPUT_S = './single_channel_100000/'

TRAIN = 'train/'
VAL = 'val/'
TEST = 'test/' 

os.makedirs(OUTPUT_S + TRAIN, exist_ok=True)

os.makedirs(OUTPUT_S + VAL, exist_ok=True)

os.makedirs(OUTPUT_S + TEST, exist_ok=True)







input_path = INPUT



Patch_size = 128
Random_Crop = 100000

names = os.listdir(input_path)
names = sorted(names)
name_s = input_path + names[0]



img_s = cv2.imread(name_s, 0)


shape = img_s.shape

Points_x = np.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_Crop)
Points_y = np.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_Crop)

count = 1
for j in range(Random_Crop):

    patch_s = img_s[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]


    if count <= 96000:
        cv2.imwrite(OUTPUT_S + TRAIN + str('%06d'%count) + '.png', patch_s)

    
    if count <= 99000 and count > 96000:
        cv2.imwrite(OUTPUT_S + VAL + str('%06d'%count) + '.png', patch_s)


    if count <= 100000 and count > 99000:
        cv2.imwrite(OUTPUT_S + TEST + str('%06d'%count) + '.png', patch_s)
  

    print("Processing image " + str('%06d'%count))
    count += 1


print('img_s', img_s.shape)

print('patch_s', patch_s.shape)





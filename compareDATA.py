import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

datasetimg = [i for i in os.listdir('./datasets/SCData/test/A')]
datasetimg.sort()

outputimg = [j for j in os.listdir('./output_img')]
outputimg.sort()

if len(datasetimg) != len(outputimg):
    print("can't make compared_img: datasetimgs and outputimgs doesn't match")
else:
    if not os.path.exists('./compared_test'):
        os.mkdir('./compared_test')


    idx = 0
    for i in tqdm(range(len(datasetimg))):
        Aimage = Image.open('./datasets/SCData/test/A/' + datasetimg[i])
        Bimage = Image.open('./datasets/SCData/test/B/' + datasetimg[i])
        GTimage = Image.open('./datasets/SCData/test/OUT/' + datasetimg[i])
        OPimage = Image.open('./output_img/'+ outputimg[i])

        upper = np.concatenate((Aimage, Bimage), axis=1)
        #print(upper.shape)
        
        lower1c = np.concatenate((GTimage, OPimage), axis=1)
        lower_expand = np.expand_dims(lower1c, axis=2)
        lower2c = np.concatenate((lower_expand, lower_expand), axis=2)
        lower3c = np.concatenate((lower2c,lower_expand), axis=2)
        #print(lower3c.shape)

        complete = np.concatenate((upper, lower3c), axis=0)
        # print(complete.shape)

        cv2.imwrite('./compared_test/' + datasetimg[i], complete)



    






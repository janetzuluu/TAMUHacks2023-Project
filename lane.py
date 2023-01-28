import cv2 
import numpy as np
image=cv2.imread('IMG_2713.jpg')
laneImage=np.copy(image)
cv2.imshow("result",image)
cv2.waitKey(0)

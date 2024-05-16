import cv2
import numpy as np



def fixSaltAndPepper(img):
   
img = cv2.imread('TC/12.png',0)



result = fixSaltAndPepper(img)

cv2.imshow('help',result)

cv2.imwrite('finalEmad.png',result)
cv2.waitKey()

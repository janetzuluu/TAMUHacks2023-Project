import cv2 
import numpy as np
def canny(image):
    gray=cv2.cvtColor(laneImage,cv2.COLOR_RGB2GRAY)
    #blur the image
    blur=cv2.GaussianBlur(gray,(5,5),0)
    canny=cv2.Canny(blur,50,150)
    return canny
def areaOfInterest(image):
    height=image.shape[0]
    polygon=np.array([
    [(200,height),(1100,height),(550,250)]])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,polygon,255)
    mask2=cv2.bitwise_and(image,mask)
    return mask2
def lineDisplay(Image,lines):
    Image_line=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape[4]
            cv2.line(Image_line,(x1,y1),(x2,y2),[255,0,0],10)
    return Image_line


image=cv2.imread('C:/Users/janet/Downloads/IMG_2720.jpg')
laneImage=np.copy(image)
canny=canny(laneImage)
new_image=areaOfInterest(canny)
line=cv2.HoughLinesP(new_image,2,np.pi/180,100, np.array([]),minLineLength=40,maxLineGap=6)
average_lines=average_slope_intercept()
Image_line=display_lines(laneImage,lines)
cv2.imshow("result",areaOfInterest(new_image))

cv2.waitKey(0)

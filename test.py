import cv2
import numpy as np

# Load the image into cv2
img = cv2.imread('C:/Users/janet/Downloads/IMG_2720.jpg')

# Convert the image to grayscale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# Area of interest of the shape we actually want to look at
height, width = edges.shape
mask = np.zeros_like(edges)
polygon = np.array([[(0, height), (width/2, height/2), (width, height)]], np.int32)
cv2.fillPoly(mask, polygon, 255)
masked_edges = cv2.bitwise_and(edges, mask)

# Apply HoughLines algo to detect lines on the picture with the masked edges
lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, maxLineGap=50)

# Split the lines into left and right lanes into arrays
leftLine = []
rightLine = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    if x1 < width/2 and x2 < width/2:
        leftLine.append(line)
    elif x1 > width/2 and x2 > width/2:
        rightLine.append(line)
# Average the lines in each lane to get a single lane line
left_average = np.mean(leftLine, axis=0).flatten().astype(int)
right_average = np.mean(rightLine, axis=0).flatten().astype(int)
cv2.line(img, (left_average[0], left_average[1]), (left_average[2], left_average[3]), (0, 255, 0), 2)
cv2.line(img, (right_average[0], right_average[1]), (right_average[2], right_average[3]), (0, 255, 0), 2)

#lane lines and show for a specificied time then close it
cv2.imshow("Sensor-I", img)
cv2.waitKey(7000)
cv2.destroyAllWindows()
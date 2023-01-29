import cv2
import numpy as np

# Load the image into cv2
img = cv2.imread('C:/Users/janet/Downloads/IMG_2720.jpg')

# Convert the image to grayscale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply gussian blur to reduce noise and make analyzing easier
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# detect edges in the picture
edges = cv2.Canny(blur, 50, 150)

# Region of interest of the shape we actually want to look at
height, width = edges.shape
mask = np.zeros_like(edges)
polygon = np.array([[(0, height), (width/2, height/2), (width, height)]], np.int32)
cv2.fillPoly(mask, polygon, 255)
masked_edges = cv2.bitwise_and(edges, mask)

# Apply HoughLinesP to detect lines
lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, maxLineGap=50)

# Split the lines into left and right lanes
left_lines = []
right_lines = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    if x1 < width/2 and x2 < width/2:
        left_lines.append(line)
    elif x1 > width/2 and x2 > width/2:
        right_lines.append(line)

# Average the lines in each lane to get a single lane line
left_avg = np.mean(left_lines, axis=0).flatten().astype(int)
right_avg = np.mean(right_lines, axis=0).flatten().astype(int)

# Draw the lane lines on the image
cv2.line(img, (left_avg[0], left_avg[1]), (left_avg[2], left_avg[3]), (0, 255, 0), 2)
cv2.line(img, (right_avg[0], right_avg[1]), (right_avg[2], right_avg[3]), (0, 255, 0), 2)

# Show the image with lane lines
cv2.imshow("Lanes", img)
cv2.waitKey(7000)
cv2.destroyAllWindows()
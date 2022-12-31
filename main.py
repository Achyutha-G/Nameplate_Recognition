import cv2
import numpy as np
import easyocr

# I have Imported ssl as I had a small error in my laptop because chrome is not the defalut browser. To bypass
# this I had to explicitly mention not to verify user certificates in chrome.

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Take the img input from the user. Change the path to that of your computer
# Use an image where the car number plate is placed in a similar position as that of the
# given images as I have not optimized the code for skewed number plates.

img = cv2.imread('/Users/achyuthag/Desktop/Nameplate_Recognition/car.jpg')

# Convert the img to grayscale as algorithms are more efficient in grayscale

convert_colour = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

im_th = cv2.threshold(convert_colour, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Apply the Component analysis function

analysis = cv2.connectedComponentsWithStats(im_th, 8, cv2.CV_32S)

# These are the values of the Component Labelling Analysis

(totalLabels, label_ids, values, centroid) = analysis

# Initialize a new image to
# store all the output components

output = np.zeros(im_th.shape, dtype="uint8")

# Loop through each component

for i in range(1, totalLabels):
    area = values[i, cv2.CC_STAT_AREA]
    if (area > 300) & (area < 9000):
        # Labels stores all the IDs of the components on the each pixel
        # It has the same dimension as the threshold
        # Check the component
        # then convert it to 255 value to mark it white

        componentMask = (label_ids == i).astype("uint8") * 255

        # Creating the Final output mask

        output = cv2.bitwise_or(output, componentMask)

# Showing the result and ocr in img form also.

cv2.imshow("Original Image", img)
cv2.imshow("Binary Image", im_th)
final_img = cv2.imshow("Filtered Components", output)

reader = easyocr.Reader(['en'])
result = reader.readtext(output, paragraph="False")
print('The result is: ', result[-1][-1])
cv2.waitKey(0)
cv2.destroyAllWindows()



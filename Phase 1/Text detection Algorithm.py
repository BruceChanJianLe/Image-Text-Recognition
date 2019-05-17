import cv2
import numpy as np
import argparse
import pytesseract


# Decoding function
def decode(scores, geometry, scoreThresh):
    detections = []
    confidence = []

    # Obtain the number of rows and columns in scores
    height, width = scores.shape[2:4]

    # loop over number of all rows
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]

        # Loop over the number of columns
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, ignore it and move to the next data / x
            if score < scoreThresh:
                continue

            # Compute the offset factor
            # Maps will be 4 times smaller than the original image
            offsetX, offsetY = x * 4.0, y * 4.0

            # Extract the rotation angle for the prediction
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cos * x1_data[x] + sin * x2_data[x], offsetY - sin * x1_data[x] + cos * x2_data[x]])

            # Find points for rectangle
            p1 = (-sin * h + offset[0], - cos * h + offset[1])
            p3 = (-cos * w + offset[0],  sin * w + offset[1])
            center = (0.5*(p1[0] + p3[0]), 0.5*(p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / np.pi))
            confidence.append(float(score))

    # Return detections and confidences
    return [detections, confidence]

# load the input image
img = cv2.imread('example_01.jpg')
ori = img.copy()
(ori_Height, ori_Width) = img.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(new_Width, new_Height) = (320, 320)
ratio_Width = ori_Width / float(new_Width)
ratio_Height = ori_Height / float(new_Height)

# resize the image and grab the new image dimensions
img = cv2.resize(img, (new_Width, new_Height))
(Height, Width) = img.shape[:2]

output_layer_name = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

net = cv2.dnn.readNet(model='frozen_east_text_detection.pb')

## NOTES FOR USING cv2.dnn.blobFromImage function. reference: https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
# The first argument is the image itself
# The second argument specifies the scaling of each pixel value. In this case, it is not required. Thus we keep it as 1.
# The default input to the network is 320×320. So, we need to specify this while creating the blob. You can experiment with any other input dimension also.
# We also specify the mean that should be subtracted from each image since this was used while training the model. The mean used is (123.68, 116.78, 103.94).
# The next argument is whether we want to swap the R and B channels. This is required since OpenCV uses BGR format and Tensorflow uses RGB format.
# The last argument is whether we want to crop the image and take the center crop. We specify False in this case.

# Blob perform 1 mean subtraction; 2 scaling; 3 channel swapping(optional) this is because OpenCV uses BGR and tensorflow uses RGB
blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(Width, Height),
                             mean=(123.68, 116.78, 103.94), swapRB=True, crop=False)

# Set the blob as input for the loaded Neural Network
net.setInput(blob)

# scores contains the probability of a given region containing text
# geometry is a map used to derive the bounding box coordinates of text region
(scores, geometry) = net.forward(output_layer_name)

# decoding the results
confThreshold = 0.5 # Threshold level
[boxes, confidences] = decode(scores, geometry, confThreshold)

# Apply NMS
nmsThreshold = 0.4
percent = 0.1
indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(boxes[i[0]])
            vertices *= [ratio_Width, ratio_Height]
            p_1, p_2 = [], []

            # find the small in row and column
            for j in range(4):
                p_1.append(vertices[j][0])
            for j in range(4):
                p_2.append(vertices[j][1])
            start_x, start_y = min(p_1), min(p_2)
            end_x, end_y = max(p_1), max(p_2)

            # Extend the ROI for better ORC
            dX = (end_x - start_x) * percent
            dY = (end_y - start_y) * percent
            print(type(dX), dY, dX / percent, dY / percent)
            #print((start_x, start_y), (end_x, end_y))
            start_x = int(start_x - dX / 2)
            end_x = int(end_x + dX / 2)
            start_y = int(start_y - dY / 2)
            end_y = int(end_y + dY / 2)
            # Draw the ROI
            cv2.rectangle(ori, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)


cv2.imshow('Detection', ori)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
import argparse
import pytesseract


# Construct the argument parser and parse the argument (for terminal use)
apr = argparse.ArgumentParser()
apr.add_argument('-i', '--image', type=str, default='example_08.jpg',
                 help='path to input image')
apr.add_argument('-east', '--east', type=str, default='frozen_east_text_detection.pb',
                 help='path to input EAST detector')
apr.add_argument('-c', '--min_confidence', type=float, default=0.5,
                 help='minimum probability as a region of interest')
apr.add_argument('-w', '--width', type=int, default=320,
                 help='nearest multiple of 32 for resize width')
apr.add_argument('-hh', '--height', type=int, default=320,
                 help='nearest multiple of 32 for resize height')
apr.add_argument('-n', '--nms', type=float, default=0.4,
                 help='percentage of extension')
apr.add_argument('-p', '--percent', type=float, default=0.0,
                 help='percentage of extension')
args = vars(apr.parse_args())


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
img = cv2.imread(args['image'])
ori = img.copy()
(ori_Height, ori_Width) = img.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
new_Width, new_Height = args['width'], args['height']
ratio_Width = ori_Width / float(new_Width)
ratio_Height = ori_Height / float(new_Height)

# resize the image and grab the new image dimensions
img = cv2.resize(img, (new_Width, new_Height))
(Height, Width) = img.shape[:2]

output_layer_name = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

# Load EAST DNN model
net = cv2.dnn.readNet(args['east'])

# Blob perform 1 mean subtraction;
# 2 scaling;
# 3 channel swapping(optional) this is because OpenCV uses BGR and tensorflow uses RGB
blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(Width, Height),
                             mean=(123.68, 116.78, 103.94), swapRB=True, crop=False)

# Set the blob as input for the loaded Neural Network
net.setInput(blob)

# scores contains the probability of a given region containing text
# geometry is a map used to derive the bounding box coordinates of text region
(scores, geometry) = net.forward(output_layer_name)

# decoding the results
[boxes, confidences] = decode(scores, geometry, args['min_confidence'])

# Apply NMS
indices = cv2.dnn.NMSBoxesRotated(boxes, confidences,
                                  args['min_confidence'], args['nms'], )

# Initialize a result list to store OCR bounding boxes
results = []
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
            dX = (end_x - start_x) * args['percent']
            dY = (end_y - start_y) * args['percent']

            # Computation for extending ROI
            start_x = int(start_x - dX / 2)
            end_x = int(end_x + dX / 2)
            start_y = int(start_y - dY / 2)
            end_y = int(end_y + dY / 2)

            if start_x < 0:
                start_x = 0
            if start_y < 0:
                start_y = 0

            # extract the region of interest
            roi = ori[start_y:end_y, start_x:end_x]

            # Using Tesseract v4
            # 1 Choose a language; 2 an OEM flag of 4, indicating that we wish to use LSTM NN model
            # 3 An OEM value, which in this case is 7, implies ROI as a single line of text
            #config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"' #'-1 eng --oem 1 --psm 7'
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            text = pytesseract.image_to_string(roi, config="-l eng --oem 1 --psm 7" )#, config=config)

            # Add the bounding box coordinate and OCR detected text to result list
            results.append(((start_x, start_y, end_x, end_y), text))

# Perform a deep copy of the original image
deep_copy = ori.copy()

# Loop over all results
for ((start_x, start_y, end_x, end_y), text) in results:
    # Display text at terminal
    print(f'The detected text:{text}')

    # Draw only the ASCII characters since putText can only work with ASCII characters
    text = ''.join([t if ord(t) < 128 else '' for t in text])

    # Draw the bounding box for ROI
    cv2.rectangle(deep_copy, (start_x, start_y), (end_x, end_y),
                  color=(0, 255, 0), thickness=2)

    # Draw the detected text, right above the bounding box
    cv2.putText(deep_copy, text, (start_x, start_y - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8, color=(0, 0, 255), thickness=2)

cv2.imshow('Detection', deep_copy)
cv2.imwrite(f"output_{args['image']}", deep_copy)

# README
## Description of algorithm and how to use it

### Phase 1

During Phase 1, I am developing an algorithm to detect the text existing in the images. The detected region will be used for Phase 2 which is text recognition. I am using 'EAST' as the text detector, which in full name it is called ' Efficient and Accurate Scene Text detector'. 

For reference: https://arxiv.org/abs/1704.03155

You will need at least OpenCV 3.4.5 or OpenCV 4 to implement the algorithm.

### Phase 2

During Phase 2, I used Tesseract v4 to obtain the text in the image and then maps the text back to the image. Do note that the process of installing Tesseract is needed.

For reference: https://stackoverflow.com/questions/51677283/tesseractnotfounderror-tesseract-is-not-installed-or-its-not-in-your-path

### Using the algorithm

I am guessing you are familiar with argparse, you can seek for more information with -h

The algorithm outputs a processed image with bounding boxes at ROI and text above it. Some ROIs are too high up the image, with makes the drawn text out of the image. But the results of detected texts are printed in the terminal.

Enjoy!

### Others

Create a new repository on the command line

echo "# Image-Text-Recognition" >> README.md

git init

git add README.md

git commit -m "first commit"

git remote add ITR https://github.com/BruceChanJianLe/Image-Text-Recognition.git

git push -u ITR master

Push an existing repository from the command line

git remote add ITR https://github.com/BruceChanJianLe/Image-Text-Recognition.git

git push -u ITR master

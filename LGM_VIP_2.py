#!/usr/bin/env python
# coding: utf-8

# # LGMVIP-DATA SCIENCE TASK-2¶
# AUTHOR: Swapnil S Yadav
# 
# Task 2:Image to Pencil Sketch with Python:
# 
# In this task , We need to read the image in RBG format and then convert it to a grayscale image. This will turn an image into a classic black and white photo. Then the next thing to do is invert the grayscale image also called negative image, this will be our inverted grayscale image. Inversion can be used to enhance details. Then we can finally create the pencil sketch by mixing the grayscale image with the inverted blurry image. This can be done by dividing the grayscale image by the inverted blurry image. Since images are just arrays, we can easily do this programmatically using the divide function from the cv2 library in Python.

# In[ ]:


import cv2

#reading the image file
img_rgb = cv2.imread("/content/Speak-like-Doremon--a-cartoon-character--women-told_1713038d0c1_medium.jpg")

#colour image to grey
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

#inverting grey image
img_gray_inv = 255 - img_gray

#applying Gaussian Blue
img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),sigmaX=0, sigmaY=0)

#applying dodge
def dodgeV2(image, mask):
  return (cv2.divide(image, 255-mask, scale=256))

#applying burn
def burnV2(image, mask):
  return (255 - cv2.divide(255-image, 255-mask, scale=256))

#blending to generate the sketch
img_blend = dodgeV2(img_gray, img_blur)

#displaying the sketch
cv2.imwrite('result.jpg',img_blend)


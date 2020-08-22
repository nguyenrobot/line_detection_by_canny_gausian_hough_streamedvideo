# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:27:06 2020

@author: nguyenrobot
"""

"""
# Line detection with Canny Filter and Hough Transform on streamed-video
    
In this tutorial, we apply the same technic based on Canny Filter and Hough Transform to detect lines.
*My previous tutorial on line-detection based on Canny Filter and Hough Transform of a static image*
https://github.com/nguyenrobot/line_detection_by_canny_gausian_hough  

This time we will try to detect lane's lines of a streamed video instead of a static image.

Our processing consist of (for the first part of this tutorial) :
- [x] Colour selection, try to keep white and yellow pixels
- [x] Gaussian filter with small kernel size to detect even blurred lines in far left/right side
- [x] Canny edge detection with small values of threshold to detect even blurred lines in far left/right side
- [x] Zone of interest filtering, to eliminate non-desired detections
- [x] Probabilistic Hough Transform with small values of minLineLength, maxLineGap and minimum_vote to be able to detect dashed-lines in far left/right side

*Author : nguyenrobot*  
*Copyright : nguyenrobot*  
https://github.com/nguyenrobot

Preprocessed-video Credit : Udacity
"""

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    frame_lines      = np.copy(img)
    frame_lines[:,:] = [0, 0, 0]
            
    frame_lines_1st = np.copy(img)
    frame_lines_1st[:,:] = [0, 0, 0]
    
    frame_lines_2nd = np.copy(img)
    frame_lines_2nd[:,:] = [0, 0, 0]    
        
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                frame_lines[y1, x1] = [255, 255, 255]
                frame_lines[y2, x2] = [255, 255, 255]
                thickness_i = 2
                cv2.line(frame_lines_1st, (x1, y1), (x2, y2), [255, 255, 255], thickness_i)

    frame_lines_1st     = grayscale(frame_lines_1st)
    # Gaussian filter
    kernel_size     = 1
    #frame_lines_1st = gaussian_blur(frame_lines_1st, kernel_size)

    # Canny edge detection
    low_threshold   = 25
    high_threshold  = 150
    #frame_lines_1st = canny(frame_lines_1st, low_threshold, high_threshold)
    
    # Hough Transform
    minLineLength   = 1
    maxLineGap      = 150
    rho             = 1
    theta           = np.pi/1440
    minimum_vote    = 100
    lines_2nd_hough = hough_lines(frame_lines_1st, rho, theta, minimum_vote, minLineLength, maxLineGap)
    
    if lines_2nd_hough is not None:
        for line in lines_2nd_hough:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    return lines #line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def color_keep_range(frame, RGB_thd):
# Define color selection threshold - keep in-range pixels
# Example : to keep white and yellow RGB_thd should be
# RGB_thd = [([200, 200, 200], [255, 255, 255]), ([150, 150, 100], [180, 180, 120])]

    color_selection_ind = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
    result              = np.copy(frame)
    
    # Define selection by color / below the threshold
    for RGB_thd_i in RGB_thd:
        color_selection_ind[:,:] = ((result[:,:,0] > RGB_thd_i[0][0]) & (result[:,:,0] < RGB_thd_i[1][0]) & \
                                    (result[:,:,1] > RGB_thd_i[0][1]) & (result[:,:,1] < RGB_thd_i[1][1]) & \
                                    (result[:,:,2] > RGB_thd_i[0][2]) & (result[:,:,2] < RGB_thd_i[1][2]))\
                                    | color_selection_ind[:,:]

    result[~color_selection_ind] = [0, 0, 0]
    return result, color_selection_ind

def color_remove_range(frame, RGB_thd):
# Define color selection threshold - remove in-range pixels
# Example : to remove white and yellow RGB_thd should be
# RGB_thd = [([200, 200, 200], [255, 255, 255]), ([150, 150, 100], [180, 180, 120])]

    color_remove_ind         = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
    color_remove_ind_temp    = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
    result                   = np.copy(frame)
    
    # Define selection by color / below the threshold
    for RGB_thd_i in RGB_thd:
        color_remove_ind_temp[:,:] = ((result[:,:,0] > RGB_thd_i[0][0]) & (result[:,:,0] < RGB_thd_i[1][0]) & \
                                    (result[:,:,1] > RGB_thd_i[0][1]) & (result[:,:,1] < RGB_thd_i[1][1]) & \
                                    (result[:,:,2] > RGB_thd_i[0][2]) & (result[:,:,2] < RGB_thd_i[1][2]))
        color_remove_ind           = color_remove_ind | color_remove_ind_temp

    result[color_remove_ind] = [0, 0, 0]
    return result, color_remove_ind

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = image
       
    # Colour Selection
    
    # keep white and yellow pixels
    RGB_thd_keep = [([150, 150, 150], [255, 255, 255]), ([160, 160, 100], [255, 210, 140])]
    result, k                          = color_keep_range(result, RGB_thd_keep)
    
    # remove unwanted colours
    RGB_thd_remove = [([0, 0, 0], [215, 215, 180])]
    result, k                          = color_remove_range(result, RGB_thd_remove)
    
    # Gray scale
    result = grayscale(result)
    
    # Gaussian filter
    kernel_size     = 13
    result = gaussian_blur(result, kernel_size)

    # Canny edge detection
    low_threshold   = 20
    high_threshold  = 30
    result = canny(result, low_threshold, high_threshold)
    
    # Zone of interest filtering
    vertices        = np.array([[(0,np.int(image.shape[0]/2) + 100), \
                      (0,image.shape[0]-1), \
                      (image.shape[1]-1,image.shape[0]-1), \
                      (image.shape[1]-1 - 0,np.int(image.shape[0]/2) + 100), \
                      (np.int(image.shape[1]/2) + 100, np.int(image.shape[0]/2) + 50), \
                      (np.int(image.shape[1]/2) - 100, np.int(image.shape[0]/2) + 50)]], \
                    dtype=np.int32)
    result = region_of_interest(result, vertices)
    
    # Hough Transform
    minLineLength   = 1
    maxLineGap      = 15
    rho             = 1
    theta           = np.pi/1440
    minimum_vote    = 30
    lines           = hough_lines(result, rho, theta, minimum_vote, minLineLength, maxLineGap)
    
    # Draw lines
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    
    # Display the result on original video
    result          = weighted_img(line_img, image)

    return result

### import video processing library
import os
os.listdir("test_videos/")
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
### ---------------------------------------------------------------------

print('Preprocessed-video Credit : Udacity')
#Load video
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1_new        = VideoFileClip("test_videos/solidWhiteRight.mp4")

#Process loaded video
white_clip_new   = clip1_new.fl_image(process_image) #NOTE: this function expects color images!!

#Write output video
white_output_new = 'test_videos_output/solidWhiteRight_out_new.mp4'
#%time white_clip_new.write_videofile(white_output_new, audio=False)
white_clip_new.write_videofile(white_output_new, audio=False)

#Display processed video
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output_new))


print('Preprocessed-video Credit : Udacity')
#Load video
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip2_new        = VideoFileClip("test_videos/solidYellowLeft.mp4")

#Process loaded video
yellow_clip_new   = clip2_new.fl_image(process_image) #NOTE: this function expects color images!!

#Write output video
yellow_output_new = 'test_videos_output/solidYellowLeft_out_new.mp4'
#%time yellow_clip_new.write_videofile(yellow_output_new, audio=False)
yellow_clip_new.write_videofile(yellow_output_new, audio=False)

#Display processed video
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output_new))


print('Preprocessed-video Credit : Udacity')
#Load video
clip3        = VideoFileClip("test_videos/challenge.mp4")

#Process loaded video
challenge_clip   = clip3.fl_image(process_image) #NOTE: this function expects color images!!

#Write output video
challenge_output = 'test_videos_output/challenge_out.mp4'
#%time challenge_clip.write_videofile(challenge_output, audio=False)
challenge_clip.write_videofile(challenge_output, audio=False)
 
#Display processed video
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))
#Imports
from cv2 import GaussianBlur
import numpy as np
import cv2

class BlurImage(object):
    '''
        Applies Gaussian Blur on the image.
    '''

    def __init__(self, radius=0):
        '''
            Arguments:
            radius (int): radius to blur
        '''

        # Write your code here
        self.radius=radius

    def __call__(self, image):
        '''
            Arguments:
            image (numpy array or PIL Image)

            Returns:
            image (numpy array or PIL Image)
        '''

        # Write your code here
        final_img=cv2.GaussianBlur(image,(5,5),self.radius,self.radius)
        return final_img







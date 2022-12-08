#Imports
import numpy as np
import cv2

class RotateImage(object):
    '''
        Rotates the image about the centre of the image.
    '''

    def __init__(self, degrees):
        '''
            Arguments:
            degrees: rotation degree.
        '''
        
        # Write your code here
        self.degrees=degrees

    def __call__(self, sample):
        '''
            Arguments:
            image (numpy array or PIL image)

            Returns:
            image (numpy array or PIL image)
        '''

        # Write your code here
        img_centre = (sample.shape[1]//2,sample.shape[0]//2)
        rmat = cv2.getRotationMatrix2D(img_centre, self.degrees, 1.0)
        final_img=cv2.warpAffine(sample,rmat,sample.shape[1::-1],cv2.INTER_LINEAR)
        return final_img
        



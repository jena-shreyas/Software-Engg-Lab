#Imports
import numpy as np
import cv2

class RescaleImage(object):
    '''
        Rescales the image to a given size.
    '''

    def __init__(self, output_size):
        '''
            Arguments:
            output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
        '''

        # Write your code here
        self.output_size=output_size

    def __call__(self, image):
        '''
            Arguments:
            image (numpy array or PIL image)

            Returns:
            image (numpy array or PIL image)

            Note: You do not need to resize the bounding boxes. ONLY RESIZE THE IMAGE.
        '''

        # Write your code here  
        h=image.shape[0]
        w=image.shape[1]

        if type(self.output_size)==tuple:
            shape=self.output_size

        elif type(self.output_size)==int:

            if w<h :
                shape=(int(h/w*self.output_size),self.output_size)
            else:
                shape=(self.output_size,int(w/h*self.output_size))    

        final_img=cv2.resize(image,shape,interpolation= cv2.INTER_LINEAR)  
        return final_img

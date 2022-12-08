#Imports
import numpy as np
import cv2

class CropImage(object):
    '''
        Performs either random cropping or center cropping.
    '''

    def __init__(self, shape, crop_type='center'):
        '''
            Arguments:
            shape: output shape of the crop (h, w)
            crop_type: center crop or random crop. Default: center
        '''

        # Write your code here
        self.shape=shape
        self.crop_type=crop_type

    def __call__(self, image):
        '''
            Arguments:
            image (numpy array or PIL image)

            Returns:
            image (numpy array or PIL image)
        '''

        # Write your code here      
        h=self.shape[1]
        w=self.shape[0]  

        if h>image.shape[0]:
            h=image.shape[0]

        if w>image.shape[1]:
            w=image.shape[1]


        if self.crop_type=='center':

            img_center=(image.shape[0]//2,image.shape[1]//2)
            x=img_center[0]-h//2
            y=img_center[1]-w//2

            final_img=image[x:x+h,y:y+w]
            return final_img

        elif self.crop_type=='random':

            x = 0 if image.shape[0]-h<=0 else np.random.randint(0,image.shape[0]-h)
            y = 0 if image.shape[1]-w<=0 else np.random.randint(0,image.shape[1]-w)

            final_img=image[x:x+h,y:y+w]
            return final_img


    





        

 
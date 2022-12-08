#Imports
import numpy as np



class FlipImage(object):
    '''
        Flips the image.
    '''

    def __init__(self, flip_type='horizontal'):
        '''
            Arguments:
            flip_type: 'horizontal' or 'vertical' Default: 'horizontal'
        '''

        # Write your code here
        self.flip_type=flip_type


    def __call__(self, image):
        '''
            Arguments:
            image (numpy array or PIL image)

            Returns:
            image (numpy array or PIL image)
        '''

        # Write your code here
        if (self.flip_type=="horizontal"):
            
            for i in range(image.shape[0]):
                for j in range(image.shape[1]//2):
                    for k in range(3):

                        temp = image[i][j][k]
                        image[i][j][k] = image[i][image.shape[1] - j -1][k]
                        image[i][image.shape[1]- j - 1][k] = temp

        elif (self.flip_type=="vertical"):
            
            for i in range(image.shape[0]//2):
                for j in range(image.shape[1]):
                    for k in range(3):

                        temp = image[i][j][k]
                        image[i][j][k] = image[image.shape[0]-1-i][j][k]
                        image[image.shape[0]-1-i][j][k]=temp

        return image                




    




       
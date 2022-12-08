#Imports
from my_package.model import InstanceSegmentationModel
from my_package.data import Dataset
from my_package.analysis import plot_visualization
from my_package.data.transforms import FlipImage, RescaleImage, BlurImage, CropImage, RotateImage
from matplotlib import pyplot as plt
import cv2

def experiment(annotation_file, segmentor, transforms, outputs):
    '''
        Function to perform the desired experiments

        Arguments:
        annotation_file: Path to annotation file
        segmentor: The image segmentor
        transforms: List of transformation classes
        outputs: path of the output folder to store the images
    '''

    #Create the instance of the dataset.
    #Iterate over all data items.
    #Get the predictions from the segmentor.
    #Draw the segmentation maps on the image and save them.
    #Do the required analysis experiments.
    
    dataset=Dataset(annotation_file,transforms)
    for i in range(len(dataset)):
        image=dataset[i]['image']
        pred_boxes,pred_masks,pred_class,pred_score = segmentor(image)
        plot_visualization(image,pred_boxes,pred_masks,pred_class,pred_score,outputs+f'Part1/{i}.jpg')

    my_img=dataset[9]['image']
    transforms = {

        'Original image': [],
        'Horizontally flipped image': [FlipImage()],
        'Blurred image' : [BlurImage()],
        'Twice Rescaled image' : [RescaleImage((2 * my_img.shape[2], 2 * my_img.shape[1]))],
        'Half Rescaled image'  : [RescaleImage((my_img.shape[2]//2,my_img.shape[1]//2))],
        '90 degree right rotated image' : [RotateImage(-90)],
        '45 degree left rotated image'  : [RotateImage(45)]
    }

    fig=plt.figure(figsize=(15,10))
    for idx,type in enumerate(transforms):

        transform=transforms[type]
        dataset.transforms=transform
        my_img_transformed=dataset[9]['image']
        pred_boxes,pred_masks,pred_class,pred_score = segmentor(my_img_transformed)
        plot_visualization(my_img_transformed,pred_boxes,pred_masks,pred_class,pred_score,outputs+f'Part2/{type}.jpg')
        plt.subplot(3,3,idx + 1, title=type)
        plt.imshow(cv2.imread(outputs + f'Part2/{type}.jpg'))

    plt.show()
    fig.savefig(outputs+f'Part2/transforms.png')    

def main():
    segmentor = InstanceSegmentationModel()
    experiment('./data/annotations.jsonl', segmentor, [], './output/') # Sample arguments to call experiment()


if __name__ == '__main__':
    main()

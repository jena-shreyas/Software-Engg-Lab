#Imports
import numpy as np
import cv2

def plot_visualization(image,bboxes,masks,classes,score,output,choice): # Write the required arguments

    # The function should plot the predicted segmentation maps and the bounding boxes on the images and save them.
    # Tip: keep the dimensions of the output image less than 800 to avoid RAM crashes.
    image=(image*255).transpose((1,2,0)).astype(np.uint8)
    clone=image.copy()
    
    for i in range(min(3,len(bboxes))):

        top=int(bboxes[i][0][1])
        bottom=int(bboxes[i][1][1])
        left=int(bboxes[i][0][0])
        right=int(bboxes[i][1][0])

        if (choice=="Segmentation"):
        
            color=np.array([np.random.randint(0,255) for i in range(3)])
            maskThreshold=0.3

            mask=masks[i]
            mask=mask.transpose((1,2,0))
            mask=cv2.resize(mask,(clone.shape[1],clone.shape[0]),interpolation=cv2.INTER_NEAREST)
            mask=(mask>maskThreshold)
            roi=clone[:,:]

            roi=roi[mask]
            blended = ((0.7 * color) + (0.3 * roi)).astype("uint8")
            clone[:,:][mask] = blended

        elif (choice=="Bounding-box"):

            top=int(bboxes[i][0][1])
            bottom=int(bboxes[i][1][1])
            left=int(bboxes[i][0][0])
            right=int(bboxes[i][1][0])
            cv2.rectangle(clone,(left,top),(right,bottom),(255,0,0),thickness=5)
            cv2.putText(clone,f"{classes[i]}: "+"{:.3f}".format(score[i]),(left,top),fontFace=2,fontScale=0.6,color=(0,255,255))
    
    cv2.imwrite(output,clone)


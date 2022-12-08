####### REQUIRED IMPORTS FROM THE PREVIOUS ASSIGNMENT #######
from pydoc import cli
from tkinter import filedialog
from src.my_package.model import InstanceSegmentationModel
from src.my_package.data import Dataset
from src.my_package.analysis import plot_visualization
from src.my_package.data.transforms import FlipImage, RescaleImage, BlurImage, CropImage, RotateImage
import PIL
from PIL import ImageTk
####### ADD THE ADDITIONAL IMPORTS FOR THIS ASSIGNMENT HERE #######
from tkinter import *
from tkinter.ttk import *
from functools import partial

filename = ""
ip_lbl = op_lbl = None
# Define the function you want to call when the filebrowser button is clicked.
def fileClick(clicked, dataset, segmentor):

	####### CODE REQUIRED (START) #######
	# This function should pop-up a dialog for the user to select an input image file.
	# Once the image is selected by the user, it should automatically get the corresponding outputs from the segmentor.
	# Hint: Call the segmentor from here, then compute the output images from using the `plot_visualization` function and save it as an image.
	# Once the output is computed it should be shown automatically based on choice the dropdown button is at.
	# To have a better clarity, please check out the sample video.
	global filename, ip_lbl
	filename = filedialog.askopenfilename(initialdir="./data/", title='Select an image file', \
		filetypes=[('JPEG files', '*.jpg'),('PNG files', '*.png')])
		
	if filename is not None:
		filename = filename[-5:]
		dict = dataset[int(filename[0])]
		print("Length of dataset :",len(dataset))
		annot_str=[data for i,data in enumerate(dataset.json_list) if i==int(filename[0])][0]
		print(annot_str)
		img = dict['image']
		print("Image shape :", img.transpose(1,2,0).shape)
		print("PNG annotation shape :", tuple(list(dict['gt_png_ann'].shape)[-2:]))
		pred_boxes,pred_masks,pred_class,pred_score = segmentor(img)
		plot_visualization(img,pred_boxes,pred_masks,pred_class,pred_score,'./output/Segmentation/'+filename,'Segmentation')
		plot_visualization(img,pred_boxes,pred_masks,pred_class,pred_score,'./output/Bounding-box/'+filename,'Bounding-box')
		e.delete(0,END)
		e.insert(0,filename)

		input_img = PIL.Image.open('./data/imgs/'+filename)	
		input_img = ImageTk.PhotoImage(input_img)
		if ip_lbl is not None:
			ip_lbl.grid_forget()
		ip_lbl = Label(root, image=input_img)
		ip_lbl.image = input_img
		ip_lbl.grid(row=1, column=0, columnspan=4)
		process(clicked)
	####### CODE REQUIRED (END) #######

# `process` function definition starts from here.
# will process the output when clicked.
def process(clicked):

	####### CODE REQUIRED (START) #######
	# Should show the corresponding segmentation or bounding boxes over the input image wrt the choice provided.
	# Note: this function will just show the output, which should have been already computed in the `fileClick` function above.
	# Note: also you should handle the case if the user clicks on the `Process` button without selecting any image file.
	if filename == "":
		print("Select a file first!")
		return
	output_img = PIL.Image.open('./output/'+clicked.get()+'/'+filename)	
	output_img = ImageTk.PhotoImage(output_img)
	global op_lbl
	if op_lbl is not None :
		op_lbl.grid_forget()
	op_lbl = Label(root, image=output_img)
	op_lbl.image = output_img
	op_lbl.grid(row=1, column=4)
	####### CODE REQUIRED (END) #######

# `main` function definition starts from here.
if __name__ == '__main__':

	####### CODE REQUIRED (START) ####### (2 lines)
	# Instantiate the root window.
	# Provide a title to the root window.
	root = Tk()
	root.title("Image Viewer")
	####### CODE REQUIRED (END) #######

	# Setting up the segmentor model.
	annotation_file = "./data/annotations.jsonl"
	transforms = []
	
	# Instantiate the segmentor model.
	segmentor = InstanceSegmentationModel()
	# Instantiate the dataset.
	dataset = Dataset(annotation_file, transforms=transforms)

	# Declare the options.
	options = ["Segmentation", "Bounding-box"]
	clicked = StringVar()
	clicked.set(options[0])
	e = Entry(root, width=70)
	e.grid(row=0, column=0)
	####### CODE REQUIRED (START) #######
	# Declare the file browsing button
	browse = Button(root, text="Browse", command=lambda: fileClick(clicked, dataset, segmentor))\
		.grid(row=0, column=1)
	####### CODE REQUIRED (END) #######

	####### CODE REQUIRED (START) #######
	# Declare the drop-down button
	option = Combobox(width=10, font=('Calibri',10), textvariable=clicked)
	option['values'] = ("Segmentation", "Bounding-box")
	option.grid(row=0, column=2)
	####### CODE REQUIRED (END) #######

	# This is a `Process` button, check out the sample video to know about its functionality
	myButton = Button(root, text="Process", command=partial(process, clicked))
	myButton.grid(row=0, column=3)
	####### CODE REQUIRED (START) ####### (1 line)
	# Execute with mainloop()
	root.mainloop()
	####### CODE REQUIRED (END) #######
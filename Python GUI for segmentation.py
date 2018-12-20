#######################################################################################################################################
#Objective: Enable user to manually segment the tree logs in an image using a GUI with priors which can be controlled by mouse keys and movements.
#######################################################################################################################################
#Functionality:
#'Left click' to select a log to segment and apply marker.
#'Mouse wheel' to control the size of the square prior
#'Right double click' to see the segmented mask
#'D' or 'd' to delete the selected log and marker
#'S' or 's' to save the segmented mask
#'N' or 'n' to open the next image
#'R' or 'r' to reset the image to the original image.
#######################################################################################################################################
#Important Note: User needs to create 4 folders and change the hardcoded folders in the program(main loop) before running it.
#1.  input_path  = '.../Train images/'                       -> Input raw images. Will be moved to output folder if segmented or to unused folder if ignored.
#2.  output_path = '.../Train images/processed/train/image/' -> The folder where the (raw) training images are stored after manual segmentation process
#3.  label_path  = '.../Train images/processed/train/label/' -> The folder where the training masks are stored after segmentation process
#4.  unused_path = '.../Train images/processed/unused/'      -> The ignored, unclean images are moved to this folder.
#######################################################################################################################################
#Author    : Dr. Alireza Aghasi
#Programmer: Praneeth Kumar Kalavai
#Date      : Dec 2018
#######################################################################################################################################
#Import the required packages
import cv2
import numpy as np
import os
import math
import shutil

#Function to calculate KL divergence between two probability distributions
def KLD(D1, D2):
    #add a very small value to avoid 'divide by zero error'
    D1+=0.00000001
    D2+=0.00000001
    #Please refer KL divergence documentation for formula
    D1oD2 = np.log(D1 / D2)
    D2oD1 = -D1oD2
    dist1 = sum(D1 * D1oD2)
    dist2 = sum(D2 * D2oD1)
    dist = dist1 + dist2
    return dist

#Function to calculate the image distribution in all the three channels (BGR).
def channel_distributions(mask):
    b, g, r = cv2.split(mask)
    #Since we added 1 while separating foreground(in divergence function), we subtract 1 to restore the image to its original form
    b_dist = b[b != 0] - 1
    g_dist = g[g != 0] - 1
    r_dist = r[r != 0] - 1
    return np.concatenate(([b_dist, g_dist, r_dist]), axis=0)

#Use the image distributions to create probability distributions. These are used in KL divergence calculation.
def prob_dist(dist):
    #get the histogram distribution and divide it into 51 bins.
    hist_dist, edges_dist = np.histogram(dist, bins=int(255 / 5))
    return hist_dist / len(dist)

#This function is called with the actual cropped image (same image in iterations until highest divergence is found) and masks (one after the other in iteration.)
#Returns the KL divergence for the image and the mask.
def divergence(crop, mask):
    #Add 1 to the cropped image. we do a bitwise and operation with mask to separate the background(zeroes) and foreground(ones)
    #Now there are no black pixels on the image.
    crop = crop + 1
    #This will take the pixels greater than 0 as foreground
    fg = cv2.bitwise_and(crop, crop, mask=mask)
    fg_d = channel_distributions(fg)
    #This will take the background
    mask2 = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(crop, crop, mask=mask2)
    bg_d = channel_distributions(bg)
    #Use the foreground and background of the image and masks to calculate divergence over the distributions
    div = KLD(prob_dist(fg_d), prob_dist(bg_d))
    return div

#Create the elliptical and circular masks.
def create_masks(start,length):
    #Local dictionaries to store circular and elliptical masks.
    #Keys would be the length of the priors and values would be the list of masks for each length
    c_dict = {}
    e_dict = {}
    #iterating from 6 to 120 in steps of 6.
    for l in range(start,length+1,6):
        #initialize the lists to store masks
        c_masks, e_masks = [],[]
        #Centre of the square with length l
        cx,cy=int(l/2),int(l/2)
        #Top left and bottom right coordinates of the square
        x11,y11,x22,y22=0,0,l,l
        #For each cropped image, when divergence is calculated using masks, the smallest masks tend to have greater divergence values.
        #To avoid getting incorrect masks, we set minimum radius of the masks based on the size of the prior.
        #Defining strides for various lengths.
        if l>=108:
            stride=6
            min_rad = 34
            step=4
        elif  90<= l < 108:
            stride = 4
            min_rad = 26
            step = 5
        elif  66<= l < 90:
            stride = 4
            min_rad = 20
            step = 6
        elif 48<=l<66:
            stride=4
            min_rad = 18
            step=5
        elif 30<=l<48:
            stride=3
            min_rad=12
            step=4
        elif 18<=l<30:
            stride = 3
            min_rad = 8
            step = 3
        elif 8 <= l < 18:
            stride = 3
            min_rad = 5
            step = 2
        else:
            stride=2
            min_rad = 2
            step=2
        for x in range(stride,l+1,stride):         #increase the x coordinate
            for y in range(stride,l+1,stride):     #increase the y coordinate
                for r in range(min_rad,int(l/2)+1,step):   #increase the radius
                    #Make sure that the circle lies within the square and avoid unwanted masks. We assume that the cursor
                    #lies on the log when user marks the log irrespective of the box size around cursor.
                    if (r < min(x-x11, y-y11, abs(x-x22),abs(y-y22))) and (int(math.sqrt((cx-x)**2+(cy-y)**2)) < int(l/4)):
                        mask = np.zeros((l, l), np.uint8)     #create a black background
                        center = (int(x), int(y))
                        cv2.circle(mask, center, r, (255, 0, 0), thickness=-1) #create a filled circle
                        c_masks.append([x,y,r,mask])                #append the mask with centre, radius details to a list
                        #Uncomment the below two to see the masks in loop. hit any key to move to next mask.
                        #cv2.imshow('image', img)
                        #cv2.waitKey()
                        #ellipse masks for the same radius sa one of the axes.
                        #for now we are going with one ratio of axes. If we need to iterate with the rations,
                        #push it to the below for loop
                        #This changes the oblongness of the ellipse
                        axes = (int(r), int(9 * r / 10))
                        #Change the angle
                        for a in range(0, 10, 2):
                            mask = np.zeros((l, l), np.uint8)
                            angle = 20 * a  # We are currently adding 40 degrees every iteration to the angle
                            cv2.ellipse(mask, center, axes, angle, startAngle=0, endAngle=360, color=(255, 0, 0),
                                        thickness=-1)
                            e_masks.append([x, y, r, angle, mask])
        #Store the masks for with length of the box as key
        c_dict[l] = c_masks
        e_dict[l] = e_masks
    return c_dict, e_dict

#Function takes the coordinates of the prior and calls necessary functions to get divergence of the cropped portion with masks.
def segmentation(coordinates):
    #refer the global dictionaries which contain masks.
    global em_dict, cm_dict
    crop = base_img[coordinates[0][1]:coordinates[1][1], coordinates[0][0]:coordinates[1][0]]
    length = coordinates[1][1]- coordinates[0][1]
    #Initialize the divergence values.
    #Max_div holds the max value in iterations over the masks. curr_div gets divergence for current mask in iteration
    max_div, curr_div = 0,0
    mx, my, mr, angle = 0, 0, 0, -1
    for mask_details in cm_dict[length]:
        curr_div = divergence(crop,mask_details[3])
        if curr_div > max_div:
            max_div=curr_div
            mx,my,mr=mask_details[:3]
    #after the above loop, max_div contains the max div details for circular masks
    for mask_details in em_dict[length]:
        curr_div = divergence(crop,mask_details[4])
        #if we get divergence greater than the circular mask, we overwrite the details with elliptical mask details
        if curr_div > max_div:
            max_div=curr_div
            mx,my,mr,angle=mask_details[:4]
    #If the angle value is greater than 0, then elliptical mask has got higher divergence.
    #Return the parameters of the mask - centre, axes, amd angle
    if angle>=0:
        return (coordinates[0][0]+mx,coordinates[0][1]+my), (int(mr), int(9 * mr / 10)), angle
    #In case of circular mask, pass centre, radius
    else:
        return (coordinates[0][0]+mx,coordinates[0][1]+my),mr

#Load the image from path and set up the image window and GUI
def load_image():
    #declare the global variables so that they can be used everywhere.
    global img, base_img, boxed_img, height,width
    #Read the first input image
    img = cv2.imread(input_path + input_images[img_num], 1)
    #img = 'C:/Users/Praveen/Desktop/GSU/MSA course/MSA Fall18/Deep learning/GP project/242712-b.jpg'
    height,width,_ = img.shape
    #Create a named window using the file name.
    cv2.namedWindow(input_images[img_num], cv2.WINDOW_FREERATIO)
    #Make a base copy of the image. We do not make any changes to this image. This is the original image/ground truth.
    base_img = img.copy()
    #boxed image is used to show markers and segmented contours on the screen.
    boxed_img = img.copy()
    #Mouse call back function to respond to any mouse actions. In case of any mouse events, it calls the function
    cv2.setMouseCallback(input_images[img_num], process_mouse_events)

#Redraw the image on GUI iteratively to reflect changes.
def redraw():
    #We copy the boxed image with new position of the rectangular prior tracked using mouse movement
    with_box_prior = boxed_img.copy()
    #Now draw the rectangle prior
    cv2.rectangle(with_box_prior, (x1, y1), (x2, y2), (255, 0, 0), 2)
    l=x2-x1
    #Draw the circle inside the rectangle prior which we use as a guidance to show the approximate minimum radius of the mask for that sized prior.
    cv2.circle(with_box_prior,(int(x1+l/2),int(y1+l/2)),int(l/4),(255, 0, 0), 2)
    cv2.imshow(input_images[img_num], with_box_prior)

def draw_markers():
    global masked, boxed_img, segmented
    #create a black canvas of the same size as the input image.
    masked = np.zeros((height, width), np.uint8)
    #get the base image replace the boxed image. Freshly mark all the logs, draw contours of circle/ellipse masks
    boxed_img = base_img.copy()
    #Draw markers on logs.
    for centre in circles_list:
        cv2.circle(boxed_img, centre, 1, (0, 0, 255), 2)
    #Draw masks on the black canvas and also contours on the tree logs in the fresh image to compare the segmentation results.
    for details in masks_list:
        if len(details) > 2:  # ellipse because it has centre, angle and axes details.
            cv2.ellipse(boxed_img, details[0], details[1], details[2], startAngle=0, endAngle=360, color=(255, 0, 0),
                        thickness=1)
            cv2.ellipse(masked, details[0], details[1], details[2], startAngle=0, endAngle=360, color=(255, 0, 0),
                        thickness=-1)
        else:
            cv2.circle(boxed_img, details[0], details[1], (255, 0, 0), thickness=1)
            cv2.circle(masked, details[0], details[1], (255, 0, 0), thickness=-1)
    #Set a flag to identify if the image was ever segmented.
    #Note that the image will be moved to a different folder based on whether it is segmented or unused.
    if len(masks_list) > 0:
        segmented = True
    redraw()

#Mouse callback function to process different mouse events.
def process_mouse_events(event,x,y,flags,param):
    #Variables
    #x1,y1,x2,y2  -> Rectangular prior coordinates,
    #rect_list    -> List of coordinates of the square portions containing the marked logs.
    #circles_list -> List of markers on each log selected.
    #masks_list   -> List of masks obtained for each log after segmentation using KL divergence. One mask for each log.
    #prev_key     -> Flag used to track double escape. The GUI closes only if user presses esc button twice without any interruptions.
    #                This is done to avoid closure of GUI when user accidentally hits escape key. This flag gets reset with any mouse or keyboard operation but Escape.
    global x1, y1, x2, y2, size, rect_list, circles_list, prev_key, masks_list,boxed_img
    if event == cv2.EVENT_MOUSEWHEEL:
        prev_key = 0
        if flags > 0 and size<60:
            size+=6  #size from 4 to 60 in steps of 6. length of the square increases by 2*size (120 max)
        elif flags <0 and size>6:
            size-=6  #decreases size upto 6.
        #calculate the top left and bottom right coordinates of the rectangular prior.
        #(x,y) is the current mouse pointer location
        x1, y1, x2, y2 = x - size, y - size, x + size, y + size
    elif event == cv2.EVENT_MOUSEMOVE:
        prev_key = 0
        #use the below if condition to restrict prior from going out of the image margins. prior freezes if it reaches image border.
        #if x - size > 0 and y - size > 0 and x + size < 1600 and y + size < 1200:
        #recalculate the current cursor position and reset the coordinates of rectangular prior
        x1,y1,x2,y2 = x-size,y-size,x+size,y+size
        #Redraw the image with latest coordinates and prior.
        redraw()
    elif event == cv2.EVENT_LBUTTONDOWN:
        #Left click
        prev_key = 0
        #print(x - size, y - size, x + size, y + size)
        #We do not mark the log if the prior goes out of the image margin.
        if x - size > 0 and y - size > 0 and x + size < 1600 and y + size < 1200:
            #Mark the log with a dot like circle in red color and store it in list. store the rectangle prior information
            rect_list.append(((x - size, y - size), (x + size, y + size)))
            circles_list.append((x,y))
            #Segmentation function called in the below code gets the mask. Append it to the masks list. Note that the rectangular prior coordinates are passed to the function.
            masks_list.append((segmentation(((x - size, y - size), (x + size, y + size)))))
            #Draw all the markers and contours.
            draw_markers()
        #redraw(boxed_img)
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        #used to confirm the selections. passes the info to next module to segment and project on a black image.
        prev_key = 0
        if segmented:
            cv2.namedWindow('masked', cv2.WINDOW_GUI_NORMAL)
            cv2.imshow('masked', masked)

if __name__ == "__main__":
    #Global coordinate and size variables. They are declared as global in process_mouse_events function.
    x1,y1,x2,y2=-1,-1,-1,-1
    #initial size of the square prior
    size=24
    #index for images in the input folder
    img_num=0
    #Initiate lists of rectangle coordinates of cropped images, circle markers on the cropped images, masks identified with maximum KL divergence
    rect_list=[]
    circles_list=[]
    masks_list = []

    #Input images exist in the below folder. Output(masked) images are stored in the output folder. Change these folders to your local folders.
    input_path = 'C:/Users/Praveen/Desktop/GSU/MSA course/MSA Fall18/Deep learning/Train images/'
    output_path = 'C:/Users/Praveen/Desktop/GSU/MSA course/MSA Fall18/Deep learning/Train images/processed/train/image/'
    label_path = 'C:/Users/Praveen/Desktop/GSU/MSA course/MSA Fall18/Deep learning/Train images/processed/train/label/'
    unused_path = 'C:/Users/Praveen/Desktop/GSU/MSA course/MSA Fall18/Deep learning/Train images/processed/unused/'

    #Get the file names of all the images in input path. The images should be in .jpg format.
    input_images = [image for image in os.listdir(input_path) if image.endswith('.jpg')]

    #Call the function to load the first image in the folder
    load_image()
    #Create masks for square priors varying in length from 6 to 120 in steps of 6
    cm_dict, em_dict = create_masks(6,120)

    #Initialize prev_key to track double escape. image closes only with double escape button to avoid accidental closure with single escape.
    prev_key = 0
    #Flag to identify if segmentation is performed on the current image.
    segmented = False
    while True:
        #Keep redrawing the image with updates(markers, contours, square prior)
        redraw()
        #Track keyboard keys. Perform intended functions as assigned.
        key = cv2.waitKey(0) & 0xFF  #gets the last 8 significant bits of the key value
        if key == 27:  #escape
            if prev_key == key:  #if previous key is escape and current key is also escape, break the loop. They should be consecutive. mouse movement is not allowed.
                prev_key = 0
                break
            prev_key = key
        elif key == (ord('d') or ord('D')):
            #Delete the markers on the image if they exist. can remove len(rect_list) since there would be one marker for each cropped image.
            if  len(circles_list) > 0 and len(rect_list) >0:
                print("Deleting the log selection and marker")
                circles_list.pop()
                rect_list.pop()
                masks_list.pop()
                draw_markers()  #draw the latest list of markers on the image after changes.
        elif key == (ord('r') or ord('R')):
            #Reset the image. Delete all markers.
            if  len(circles_list) > 0 and len(rect_list) >0:
                print("Resetting the image")
                circles_list=[]
                rect_list=[]
                masks_list=[]
                draw_markers()
        elif key == (ord('s') or ord('S')):
            #Save the segmented image only if the image is segmented.
            if segmented:
                cv2.imwrite(os.path.join(label_path,input_images[img_num]),masked)
        elif key == (ord('n') or ord('N')):
            #Load next image.
            circles_list = []
            rect_list = []
            masks_list = []
            x1, y1, x2, y2 = -1, -1, -1, -1
            cv2.destroyWindow(input_images[img_num])
            cv2.destroyWindow('masked')
            #move the previous image to the output folder.
            if segmented:
                shutil.move(os.path.join(input_path,input_images[img_num]),os.path.join(output_path,input_images[img_num]))
            else:
                shutil.move(os.path.join(input_path,input_images[img_num]),os.path.join(unused_path, input_images[img_num]))
            img_num+=1
            segmented = False
            if img_num <= len(input_images):
                load_image()
    cv2.destroyAllWindows()  #Destroy all windows once out of loop.

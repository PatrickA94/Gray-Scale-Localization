import re
import glob
import os.path
import argparse
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold='nan')
import scipy.ndimage
from PIL import Image, ImageDraw
import cv2


def compare_images(frame1dir, frame2dir):
    if not frame1dir or not frame2dir: print "compare_image :: Bad image detected"; return False
    frame1 =(cv2.imread(frame1dir))
    frame2 = (cv2.imread(frame2dir))

#Calculating useful image data
    if len(frame1) == len(frame2):

        total_pixels = len(frame1)
        pixel_rows = frame1.shape[0]
        pixel_columns = frame1.shape[1]
        image_increment = int(round(float(total_pixels) / 90))

    else:
        print "frame size mismatch"
        return False;

    #Base image to augment
    zeroray = np.zeros((pixel_rows,pixel_columns), dtype=np.bool)

    #Pixel checking loops
    for i in range(1,pixel_rows-1,3):
        for j in range(1,pixel_columns-1,3):
    #Mean condition calculation by taking mean values of 3x3 pixels in each frame
            mean_condition= ((abs(frame2.item(i - 1, j,0) - frame1.item(i - 1, j,0)) + abs(
            frame2.item(i, j + 1,0) - frame1.item(i, j + 1,0)) +
            abs(frame2.item(i , j-1,0) - frame1.item(i, j-1,0)) + abs(frame2.item(i + 1, j,0) - frame1.item(i + 1, j,0)) + (abs(frame2.item(i , j,0) - frame1.item(i, j,0))
            )))/5

            #checking if mean condition at this location passes threshold, if it does mark a pixel box
            if (mean_condition > 15+image_increment):

                zeroray.itemset((i+1, j + 1), 1)
                zeroray.itemset((i, j-1), 1)
                zeroray.itemset((i -1, j), 1)
                zeroray.itemset((i , j + 1), 1)
                zeroray.itemset((i + 1, j), 1)
                zeroray.itemset((i , j), 1)
                zeroray.itemset((i - 1, j - 1), 1)
                zeroray.itemset((i - 1, j + 1), 1)
                zeroray.itemset((i + 1 , j - 1), 1)

    return zeroray


class Component_grouping():

    def __init__(self):
        self.parent = []
        self.label=0
        for i in xrange(0,1000):
            self.parent.append(0)

    def find(self,labelval,parent):
        self.label = labelval
        self.label2 = labelval

        while parent[self.label] != 0:
            self.label = parent[self.label]

        if (self.label != self.label2 ):
            compress = parent[self.label2]
            while(compress <> self.label):
                parent[self.label2] = self.label
                self.label2 = compress
                compress = parent[compress]

        return self.label

    def union(self,set1,set2,parent):
        self.set1 = set1
        self.set2 = set2
        while parent[self.set1] != 0:
            self.set1 = parent[self.set1]
        while parent[self.set2] != 0:
            self.set2 = parent[self.set2]
        if self.set1 != self.set2:
            parent[self.set2]=self.set1

    def prior_neighbors(self,y_cord,x_cord,bin_img):
        self.y_cord=y_cord
        self.x_cord=x_cord
        self.bin_img=bin_img
        self.pixel = []
        self.pixelval1 = [y_cord-1,x_cord-1]
        self.pixelval2 = [y_cord-1,x_cord]
        self.pixelval3 = [y_cord-1,x_cord+1]
        self.pixelval4 = [y_cord,x_cord-1]


        self.pixelval = [self.pixelval1,self.pixelval2,self.pixelval3,self.pixelval4]

        for x in self.pixelval:
            if (0<= x[0]<self.bin_img.shape[0]) and (0<= x[1] < self.bin_img.shape[1]):
                if self.bin_img[x[0],x[1]] ==1:
                    self.pixel.append(x)
        return self.pixel

    def labels(self,prior_neigh,component_img):
        self.prior_neigh=prior_neigh
        self.comp_img=component_img
        self.labelArray = []
        for neighbor in self.prior_neigh:
            x = neighbor[0]
            y = neighbor[1]
            self.labelArray.append(self.comp_img[x,y])
        return self.labelArray

    def group_comp(self,bin_img,component_img,height,width):
        self.bin_img=bin_img
        self.component_img = component_img
        for y in xrange(0,height):
            for x in xrange(0,width):
                if self.bin_img[y,x] ==1:
                    pix_loc_arr_neigh = self.prior_neighbors(y,x,self.bin_img)
                    if not pix_loc_arr_neigh:
                        M = self.label
                        self.label = self.label +1
                    else:
                        M = min(self.labels(pix_loc_arr_neigh, self.component_img))
                    self.component_img[y,x] = M
                    for label in self.labels(pix_loc_arr_neigh,self.component_img):
                        if label<>M:
                            self.union(M,label,self.parent)
        for y in xrange(0,height):
            for x in xrange(0,width):
                if self.bin_img.item(y,x) ==1:
                    self.component_img[y,x] = self.find(self.component_img[y,x],self.parent)
        return self.component_img


class post_Proc():

    def __init__(self, component_img,):
        self.component_img = component_img

    def sig_components(self,size):
        self.size =size
        self.component_img = self.component_img.astype(np.uint8)
        self.component_img_flat = self.component_img.flatten()

        hist, bins = np.histogram(self.component_img_flat, bins=255, range=(1, 255))
        hist[np.where(hist <= self.size)] = 0

        sig_comp = np.where(hist != 0)[0]
        sig_comp = np.add(sig_comp, 1)

        return sig_comp

    def comp_size_filt(self,size):
        self.size = size
        self.component_img = self.component_img.astype(np.uint8)
        self.component_img_flat = self.component_img.flatten()

        hist,bins = np.histogram(self.component_img_flat,bins=255, range=(1,255))
        hist[np.where(hist<=self.size)]=0
        self.component_img = np.reshape(self.component_img_flat,(self.component_img.shape[0],self.component_img.shape[1]))

        return self.component_img

    def box(self, sig_comp, component_img,image):
        self.image = Image.fromarray(image)
        self.sig_comp = sig_comp
        self.component_img = component_img
        dr = ImageDraw.Draw(self.image)

        for comp in sig_comp:
            pixelspace=np.where(self.component_img == comp)
            max_x=0
            min_x=9999
            max_y=0
            min_y=9999

            for pixel in zip(*pixelspace[::-1]):
                if pixel[0]>max_x:
                    max_x=pixel[0]
                if pixel[0]<min_x:
                    min_x=pixel[0]
                if pixel[1]>max_y:
                    max_y=pixel[1]
                if pixel[1]<min_y:
                    min_y=pixel[1]
            dr.rectangle((min_x,min_y,max_x,max_y),outline="black")
        return self.image

def imageLoop():
    # image file interaction setting paths to variables etc.
    ap = argparse.ArgumentParser()
    ap.add_argument("-ext", "--extension", required=False, default='jpg', help="extension name. default is 'png'.")
    ap.add_argument("-o", "--output", required=False, default='ex1.avi', help="output video file")
    args = vars(ap.parse_args())

    # dir_path = 'C:/Users/100594283/Desktop/moving-localization/exercise-6/'
    #dir_path = 'C:/Users/100505533/Google Drive/UOIT/Software Engineering/Second Year(2016_2017)/Winter Semester/Data Structures/Project/project-exercises/moving-localization/exercise-6'
    dir_path = 'D:\Google Drive\UOIT\Software Engineering\Second Year(2016_2017)\Winter Semester\Data Structures\Project\project-exercises\moving-localization\exercise-5'
    ext = args['extension']
    output = args['output']

    numbers = re.compile(r'(\d+)')

    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    images = []
    for filename in sorted(glob.glob("D:\Google Drive\UOIT\Software Engineering\Second Year(2016_2017)\Winter Semester\Data Structures\Project\project-exercises\moving-localization\exercise-5\*.jpg"), key=numericalSort):
         images.append(filename)


    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])

    regular_frame = cv2.imread(image_path)

    height, width, channels = regular_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(output, fourcc, 5.0, (width, height))

    # image opening loop

    for image in range(0, 90, 1):

        x = compare_images(images[image],images[image+1])
        grp = Component_grouping()
        grp = grp.group_comp(x, np.empty([height, width], dtype=int),height, width)
        #grp2 = Image.fromarray(grp.astype(np.uint8))
        #grp2.save('D:\Google Drive\UOIT\Software Engineering\Second Year(2016_2017)\Winter Semester\Data Structures\Project\Diff images\diffimg'+str(image)+'.jpg')
        img = Image.open(images[image])  # Import the Image
        img_array = np.array(img)  # convert to a array

        new = post_Proc(grp)
        sig = new.sig_components(0)
        d = new.box(sig, grp, img_array)
        d = np.array(d)
        # out.write(regular_frame)  # Write out frame to video

        cv2.imshow('video', d)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(output))

imageLoop()
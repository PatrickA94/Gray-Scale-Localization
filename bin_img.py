import re
import glob
import os.path
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold='nan')
import scipy.ndimage
from PIL import Image, ImageDraw


class Pictures():


    def __init__(self,image):
        self.img = Image.open(image).convert('L')#Import the Image
        self.img_array = np.array(self.img)
        self.filt_img_array=self.filtered_Img(3)#NEED MAKE FUNCTION FOR GAUS
        self.filt_img_array= self.image_histogram_equalization(self.filt_img_array)
        self.imgshape = self.img_array.shape




#Need function to decide gaus filter




    def filtered_Img(self,gaus_filt):
        self.img_array_int = self.img_array.astype(int)
        self.filt_img_array = scipy.ndimage.filters.gaussian_filter(self.img_array_int,gaus_filt)

        return self.filt_img_array

    def histogram(self):
        self.hist, self.bins = np.histogram(self.img_array, bins=255, range=(1, 255))
        return self.hist

    def image_histogram_equalization(self, image, number_bins=256):

        # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

        # get image histogram
        image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
        cdf = image_histogram.cumsum()  # cumulative distribution function
        cdf = 255 * cdf / cdf[-1]  # normalize

        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        return image_equalized.reshape(image.shape)

    def imgrt(self):
        return self.filt_img_array

    def shape(self,x):
        return self.imgshape[x]

    def array_return(self):
        return self.img_array


class compare():

    def __init__(self,first,second):
        self.first=first
        self.second=second





#Ceate function for thresh



    def bin_diff(self,thresh):
        self.diff = abs(self.first.imgrt().__sub__(self.second.imgrt()))
        self.diff2 = abs(self.second.imgrt().__sub__(self.first.imgrt()))

        self.diff = self.diff.flatten()
        self.diff2 = self.diff2.flatten()

        for pixel in np.nditer(self.diff,op_flags=['readwrite']):
            if pixel < thresh:
                pixel[...] = 0
            else:
                pixel[...] = 1

        for pixel1 in np.nditer(self.diff2, op_flags=['readwrite']):
            if pixel1 < thresh:
                pixel1[...] = 0
            else:
                pixel1[...] = 1

        self.diff = np.logical_and(self.diff, self.diff2)

        self.diff = self.diff.astype(int)


        x=self.first.shape(0)
        y= self.first.shape(1)
        self.diff = np.reshape(self.diff,(x,y))

        return self.diff

class Component_grouping():

    def __init__(self):
        self.parent = []
        self.label=0
        for i in xrange(0,10000):
            self.parent.append(0)

    def find(self,labelval,parent):
        self.label = labelval
        while parent[self.label] != 0:
            self.label=parent[self.label]
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

        #Adding more pixels to get better grouping
        self.pixelval5 = [y_cord-2,x_cord]
        self.pixelval6 = [y_cord-3,x_cord]
        self.pixelval7 = [y_cord-4,x_cord]
        self.pixelval8 = [y_cord,x_cord-2]
        self.pixelval9 = [y_cord,x_cord-3]
        self.pixelval10 = [y_cord,x_cord-4]



        self.pixelval = [self.pixelval1,self.pixelval2,self.pixelval3,self.pixelval4,self.pixelval5,self.pixelval6,self.pixelval7,self.pixelval8,self.pixelval9,self.pixelval10,]
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

    def group_comp(self,bin_img,component_img):
        self.bin_img=bin_img
        self.component_img = component_img
        for y in xrange(0,self.bin_img.shape[0]):
            for x in xrange(0,self.bin_img.shape[1]):
                self.component_img[y,x] = 0
            for x in xrange(0,self.bin_img.shape[1]):
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
        for y in xrange(0,self.bin_img.shape[0]):
            for x in xrange(0,self.bin_img.shape[1]):
                if self.bin_img[y,x] ==1:
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

        notsig_comp = np.where(hist == 0)[0]
        notsig_comp = np.add(notsig_comp,1)
        notsig_comp = np.insert(notsig_comp,0,0)


        for x in np.nditer(self.component_img_flat, op_flags=['readwrite']):
            if x in notsig_comp:
                x[...]=0

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





numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

frame = []
for filename in sorted(glob.glob("*.jpg"), key=numericalSort):
    frame.append(Pictures(filename))

"""
averageback = frame[0].array_return()
averageback = averageback.__add__(frame[1].array_return())
averageback = averageback.__div__(2)



for x in frame:

    averageback= averageback.__add__(frame[1].array_return())
    averageback = averageback.__div__(2)
    count = count +1

averageback=averageback.astype(np.uint8)
averageback = Image.fromarray(averageback)
averageback.save("average.jpg")

"""
count = 1
for x in frame:
    comp = compare(frame[count-1], frame[count])
    comp = comp.bin_diff(30)
    grp =Component_grouping()
    grp = grp.group_comp(comp,np.empty([comp.shape[0],comp.shape[1]], dtype= int) )
    grp2 = Image.fromarray(grp.astype(np.uint8))
    grp2.save(
        'D:\Google Drive\UOIT\Software Engineering\Second Year(2016_2017)\Winter Semester\Data Structures\Project\Computer vision\CV-Project\cv\output\diffimg' + str(count) + ".jpg")
    new = post_Proc(grp)
    sig = new.sig_components(200)
    compim = new.comp_size_filt(200)
    d = new.box(sig,compim,frame[count].array_return())


    d.save(
        'D:\Google Drive\UOIT\Software Engineering\Second Year(2016_2017)\Winter Semester\Data Structures\Project\Computer vision\CV-Project\cv\output\pic' + str(count) + ".jpg")

    count = count + 1

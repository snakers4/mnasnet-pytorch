import random
import numpy as np
from skimage import draw
from skimage.morphology import thin
# from pipeline2d3d.src.util.Util import min_max
import cv2
from skimage.morphology import disk,square
from skimage.morphology import binary_dilation
from skimage.segmentation import find_boundaries

cv2.setNumThreads(0)

class BboxMutations(object):
    def __init__(self,
                 mutations=[],
                 thin_iterations = 0, # thin iterations is exclusive with binary_dilation_disk_size
                 binary_dilation_disk_size = 0, # only elli is supported
                 find_boundaries = False
                ):
        self.mutations = mutations
        self.thin_iterations = thin_iterations
        self.binary_dilation_disk_size = binary_dilation_disk_size
        self.t = random.choice(self.mutations)
        self.find_boundaries = find_boundaries
        # print('{} is activated'.format(self.t))        
        self.mutation_dict = {
            'rect':self.rectangle_mutation,
            'elli':self.ellipse_mutation,
            'dlte':self.dilate_mutation,
            'thrs':self.ths_mutation,
            'dlto':self.dilate_open_mutation,
            'ellr':self.ellipse_mutation_rotate,
        }
        self.rotate_dict = {}
        for i in range(0,32):
            self.rotate_dict[i+1] = -3.14*(i-8)/8                
        
    def __call__(self,
                 img,
                 msk,
                 bbox):
        msk = self.mutation_dict[self.t](img,msk,bbox)
        return msk
    def rectangle_mutation(self,img,msk,bbox):
        start = (bbox.y_min,bbox.x_min)
        end  = (bbox.y_max,bbox.x_max)
        rr, cc = draw.rectangle(start, end=end , shape=msk.shape)
        
        if self.thin_iterations == 0:
            if self.binary_dilation_disk_size > 0:
                # do not dilate what we already have on the image
                tmp_msk = np.zeros_like(msk)
                tmp_msk[rr, cc] = 1
                tmp_msk = self.binary_dilation_disk(tmp_msk,self.binary_dilation_disk_size)
                msk = msk + tmp_msk
            elif self.find_boundaries == True:
                tmp_msk = np.zeros_like(msk)
                tmp_msk[rr, cc] = 1                
                border_msk = find_boundaries(tmp_msk, mode='inner', background=0)
                
                # selem = square(3)
                # border_msk = binary_dilation(border_msk,selem)                
                
                msk = msk + tmp_msk + border_msk.astype('uint8')
            else:
                msk[rr, cc] += 1
            return msk
        else:
            tmp_msk = np.zeros_like(msk)
            tmp_msk[rr, cc] = 1 
            tmp_msk = self.thin_region_fast(tmp_msk,self.thin_iterations)
            msk += tmp_msk
            return msk
    def ellipse_mutation(self,img,msk,bbox):
        c = (bbox.x_max+bbox.x_min)/2
        r = (bbox.y_max+bbox.y_min)/2
        c_radius = (bbox.x_max-bbox.x_min)/2
        r_radius = (bbox.y_max-bbox.y_min)/2
        rr, cc = draw.ellipse(r, c, r_radius, c_radius, shape=msk.shape, rotation=0.0)
        if self.thin_iterations == 0:
            if self.binary_dilation_disk_size > 0:
                # do not dilate what we already have on the image
                tmp_msk = np.zeros_like(msk)
                tmp_msk[rr, cc] = 1
                tmp_msk = self.binary_dilation_disk(tmp_msk,self.binary_dilation_disk_size)
                msk = msk + tmp_msk
            elif self.find_boundaries == True:
                tmp_msk = np.zeros_like(msk)
                tmp_msk[rr, cc] = 1                
                border_msk = find_boundaries(tmp_msk, mode='inner', background=0)
                
                msk = msk + tmp_msk + border_msk.astype('uint8')
            else:
                msk[rr, cc] += 1
            return msk
        else:
            tmp_msk = np.zeros_like(msk)
            tmp_msk[rr, cc] = 1
            # more reduction for large teeth
            if tmp_msk.sum()<500:
                tmp_msk = self.thin_region_fast(tmp_msk,self.thin_iterations)
            else:
                # print('Higher ths activated')
                tmp_msk = self.thin_region_fast(tmp_msk,int(self.thin_iterations*2))
            msk += tmp_msk
            return msk
    def ellipse_mutation_rotate(self,img,msk,bbox):
        c = (bbox.x_max+bbox.x_min)/2
        r = (bbox.y_max+bbox.y_min)/2
        c_radius = (bbox.x_max-bbox.x_min)/2
        r_radius = (bbox.y_max-bbox.y_min)/2
        rr, cc = draw.ellipse(r, c, r_radius, c_radius, shape=None, rotation=self.rotate_dict[bbox.tooth_number])
        if self.thin_iterations == 0:
            msk[rr, cc] += 1
            return msk
        else:
            tmp_msk = np.zeros_like(msk)
            tmp_msk[rr, cc] = 1 
            tmp_msk = self.thin_region_fast(tmp_msk,self.thin_iterations)
            msk += tmp_msk
            return msk   
    def dilate_mutation(self,img,msk,bbox):
        img_tooth = np.copy(min_max(img[bbox.y_min:bbox.y_max,bbox.x_min:bbox.x_max]))
        img_tooth = (img_tooth*255).astype('uint8')
        ret, thresh = cv2.threshold(img_tooth,0,255,cv2.THRESH_OTSU)
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=4)         

        if self.thin_iterations == 0:
            msk[bbox.y_min:bbox.y_max,bbox.x_min:bbox.x_max] += sure_bg
            return msk
        else:
            tmp_msk = sure_bg
            tmp_msk = self.thin_region_fast(tmp_msk,self.thin_iterations)
            msk[bbox.y_min:bbox.y_max,bbox.x_min:bbox.x_max] += tmp_msk
            return msk        
    def dilate_open_mutation(self,img,msk,bbox):
        img_tooth = np.copy(min_max(img[bbox.y_min:bbox.y_max,bbox.x_min:bbox.x_max]))
        img_tooth = (img_tooth*255).astype('uint8')
        ret, thresh = cv2.threshold(img_tooth,0,255,cv2.THRESH_OTSU)
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=4)
        opening = cv2.morphologyEx(sure_bg,cv2.MORPH_OPEN,kernel, iterations = 2)
        
        if self.thin_iterations == 0:
            msk[bbox.y_min:bbox.y_max,bbox.x_min:bbox.x_max] += opening
            return msk
        else:
            tmp_msk = opening
            tmp_msk = self.thin_region_fast(tmp_msk,self.thin_iterations)
            msk[bbox.y_min:bbox.y_max,bbox.x_min:bbox.x_max] += tmp_msk
            return msk
    def ths_mutation(self,img,msk,bbox):
        img_tooth = np.copy(min_max(img[bbox.y_min:bbox.y_max,bbox.x_min:bbox.x_max]))
        img_tooth = (img_tooth*255).astype('uint8')
        
        ret, thresh = cv2.threshold(img_tooth,0,255,cv2.THRESH_OTSU)
        msk[bbox.y_min:bbox.y_max,bbox.x_min:bbox.x_max] = thresh       
        return msk
    def thin_region_fast(self,mask,iterations):
        if mask.sum()==0:
            return mask
        else:
            min_x, max_x = np.argwhere(mask > 0)[:,0].min(),np.argwhere(mask > 0)[:,0].max()
            min_y, max_y = np.argwhere(mask > 0)[:,1].min(),np.argwhere(mask > 0)[:,1].max()

            empty = np.zeros_like(mask)
            try:
                empty[min_x:max_x,min_y:max_y] = thin(mask[min_x:max_x,min_y:max_y],max_iter=iterations)
                return empty
            except:
                return empty
            
    def binary_dilation_disk(self,mask,disk_size):
        if mask.sum()==0:
            return mask
        else:
            selem = disk(disk_size)
            return binary_dilation(mask,selem)

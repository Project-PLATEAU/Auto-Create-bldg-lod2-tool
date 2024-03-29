import cv2
import numpy as np
import random
import torch

class ImageTransformer:
    @staticmethod
    def mean(img):
        h, w, ch = img.shape
        conv = np.zeros(shape=(h, w, 1), dtype=np.float32)
        for c in range(ch):
            conv[:,:,0] += img[:,:,c]
        conv /= ch
        
        return conv.astype(np.uint8)

    @staticmethod
    def resize(img, height, width, method=cv2.INTER_CUBIC):
        h, w, ch = img.shape
        conv = np.zeros(shape=(height, width, ch), dtype=np.uint8)
        for c in range(ch):
            conv[:,:,c] = cv2.resize(img[:,:,c], (width, height) , interpolation=method)
        
        return conv

    @staticmethod
    def make_power_2(img, base, method=cv2.INTER_CUBIC):
        oh, ow, ch = img.shape
        h = int(round(oh / base) * base)
        w = int(round(ow / base) * base)
        if (h == oh) and (w == ow):
            return img

        print_size_warning(ow, oh, w, h)
        conv = np.zeros(shape=(h, w, ch), dtype=np.uint8)
        for c in range(ch):
            conv[:,:,c] = cv2.resize(img[:,:,c], (h, w) , interpolation=method)
        
        return conv

    @staticmethod
    def scale_width(img, target_width, method=cv2.INTER_CUBIC):
        oh, ow, ch = img.shape
        if (ow == target_width):
            return img
        w = target_width
        h = int(target_width * oh / ow)
        conv = np.zeros(shape=(h, w, ch), dtype=np.uint8)
        for c in range(ch):
            conv[:,:,c] = cv2.resize(img[:,:,c], (h, w) , interpolation=method)
        
        return conv

    @staticmethod
    def random_crop(img, size):
        h, w, ch = img.shape
        if w == size and h == size:
            return img
    
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)

        return img[y:y+size,x:x+size,:]

    @staticmethod
    def crop(img, pos, size):
        h, w, ch = img.shape
        x, y = pos
        if x + size > w or y + size > h:
            return img
    
        return img[y:y+size,x:x+size,:]

    @staticmethod
    def random_horizontal_flip(img):
        if random.random() <= 0.5:
            return img
    
        h, w, ch = img.shape
        conv = np.zeros(shape=(h, w, ch), dtype=np.uint8)
        for c in range(ch):
            conv[:,:,c] = np.fliplr(img[:,:,c])
    
        return conv

    @staticmethod
    def flip(img, flip):
        if flip == False:
            return img

        h, w, ch = img.shape
        conv = np.zeros(shape=(h, w, ch), dtype=np.uint8)
        for c in range(ch):
            conv[:,:,c] = np.fliplr(img[:,:,c])
    
        return conv

    @staticmethod
    def normalize(img, mean, std):
        img = np.transpose(img, (2, 0, 1))
        img = (np.asanyarray(img, dtype=np.float32) / 255.0 - mean) / std
        return torch.from_numpy(img).type(torch.FloatTensor)

    @staticmethod
    def print_size_warning(ow, oh, w, h):
        """Print warning information about image size(only print once)"""
        if not hasattr(print_size_warning, 'has_printed'):
            print("The image size needs to be a multiple of 4. "
                "The loaded image size was (%d, %d), so it was adjusted to "
                "(%d, %d). This adjustment will be done to all images "
                "whose sizes are not multiples of 4" % (ow, oh, w, h))
            print_size_warning.has_printed = True

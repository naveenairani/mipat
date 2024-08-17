import numpy as np
import cv2 as cv

class DisplayTumor:
    def __init__(self):
        self.curImg = None
        self.Img = None
        self.thresh = None
        self.kernel = np.ones((3, 3), np.uint8)

    def readImage(self, img):
        self.Img = np.array(img)
        self.curImg = np.array(img)
        gray = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)  # Ensure conversion from RGB
        self.ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    def getImage(self):
        if self.curImg is None:
            raise ValueError("Current image not set. Call readImage first.")
        return self.curImg

    # noise removal
    def removeNoise(self):
        if self.thresh is None:
            raise ValueError("Threshold image not set. Call readImage first.")
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, self.kernel, iterations=2)
        self.curImg = opening

    def displayTumor(self):
        if self.curImg is None:
            raise ValueError("Current image not set. Call readImage first.")
        sure_bg = cv.dilate(self.curImg, self.kernel, iterations=3)
        dist_transform = cv.distanceTransform(self.curImg, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        ret, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv.watershed(self.Img, markers)
        self.Img[markers == -1] = [255, 0, 0]
        tumorImage = cv.cvtColor(self.Img, cv.COLOR_HSV2BGR)
        self.curImg = tumorImage

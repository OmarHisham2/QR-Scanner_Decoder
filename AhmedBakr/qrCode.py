import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import utils as ut

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point({self.x}, {self.y})"

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

class QrCode:
    def __init__(self):
        pass

    def generate_qr_code(self, data, file_name):
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img.save(file_name)

    def read_qr_code(self, file_name):
        from pyzbar.pyzbar import decode
        from PIL import Image

        img = Image.open(file_name)
        result = decode(img)
        return result[0].data.decode("utf-8") 

def locatorContours(image,maxCorners=500,minDistance=6,showImages=False, qualityLevel=0.25,invertColors=False):
    contours=getSquares(image=image,maxCorners=maxCorners,minDistance=minDistance,showImages=showImages,qualityLevel=qualityLevel, invertColors=invertColors)
    for contour in contours:
        
    

    return new_contours

def getSquares(image,maxCorners=500,minDistance=6,showImages=False, qualityLevel=0.25, invertColors=False):
    """ 
    takes the image and returns the locator boxes contours.
    
    Parameters
    ----------
    image : colored image object.
    maxCorners : int, optional
        The maximum number of corners to return. The default is 500.
    minDistance : int, optional
        parameter to the error between contours and detected corners. Default is 10.
    showImages : bool, optional
        display contours and corners and new contours images.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    if invertColors is True:
        thresh = cv2.bitwise_not(thresh)
    
    corners = cv2.goodFeaturesToTrack(thresh, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance)
    corners = np.int0(corners)
    
    contours,heirarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    corners=list(corners)
    if showImages is True:
        originalCorners= corners.copy()
    new_contours = []  # List to store new contours
    flag=False
    for contour in contours:
    # List to store new contour points
        new_contour_points = []

        # Iterate over hull points
        for contourPoint in contour:
            if len(new_contour_points) >4:
                flag=True
                break

            x, y = contourPoint.ravel()
            # Check if hull point is close to any corner
            for i,corner in enumerate(corners):
                cx, cy = corner.ravel()
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)

                # If hull point is close to a corner, add it to new contour points
                if dist < 5:  # Adjust threshold as needed
                    new_contour_points.append(np.array([np.array([cx,cy])]))  #[[x,y],[x,y]]
                    del corners[i]
                    break
        if flag is True:    # if more than 4 corners are found, then the contour isn't a locator box
            continue
        new_contours.append(new_contour_points)

    new_contours = [contour for contour in new_contours if len(contour) > 2]  # Filter out contours with less than 3 points, which might be useless
    
    potential_boxes = ([])
    for contour in new_contours:
        x,y,w,h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if abs(aspect_ratio - 1) > 0.1: # Aspect ratio should be close to 1 if it is a square-like or rectangular shape
            continue # Not what we are looking for.
        min_size = min(w, h)
        if min_size < 10:  # Img_width [Should] be the same for all images.          
            continue                    

        potential_boxes.append(np.array(contour))
    
    if showImages is True:
        _, axarr = plt.subplots(nrows=1, ncols=3, figsize=(10,5)) # figsize is in inches, yuck
        plt.sca(axarr[0]); plt.title('Contour Points'); plt.imshow(drawContourPoints(image.copy(),contours));
        plt.sca(axarr[1]); plt.title('corner Points'); plt.imshow(drawCornerPoints(image.copy(),originalCorners));
        plt.sca(axarr[2]); plt.title('new contour points'); plt.imshow(drawContourPoints(image.copy(),new_contours));
        plt.show()

    return np.array(potential_boxes)

def getLocatorBoxes(squares):
    equal_center_groups={}
    for contour in squares:   # group centers if a point has more than one square, then it is a locator box
        M1 = cv2.moments(contour)
        cx1 = int(M1["m10"] / M1["m00"])
        cy1 = int(M1["m01"] / M1["m00"])

        similar_center_found = False
        for existing_center in equal_center_groups.keys():
            distance = np.sqrt((cx1 - existing_center[0])**2 + (cy1 - existing_center[1])**2)
            if abs(distance) <= 4:
                similar_center_found = True
                equal_center_groups[existing_center].append(contour)
                break
        
        if similar_center_found is False:
            equal_center_groups[(cx1, cy1)].append(contour)

    #remove the squares that dont have another square inside with the same center
    keys_to_remove = [key for key, value in equal_center_groups.items() if len(value) < 2]
    for key in keys_to_remove:
        del equal_center_groups[key]
        
    for contours in equal_center_groups.values():   # sort nested contours by area in descending order
        sorted(contours, key=cv2.contourArea, reverse=True)

    # Calculate distances between each pair of square contours
    distances = {}
    centerPoints=equal_center_groups.keys()
    for i in range(len(centerPoints)):
        for j in range(i+1, len(centerPoints)):
            if i != j:
                # Calculate distance between centroids of square contours
                M1 = cv2.moments(centerPoints[i])
                M2 = cv2.moments(centerPoints[j])
                cx1 = int(M1["m10"] / M1["m00"])
                cy1 = int(M1["m01"] / M1["m00"])
                cx2 = int(M2["m10"] / M2["m00"])
                cy2 = int(M2["m01"] / M2["m00"])
                distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
                contourWidth= np.sqrt(cv2.contourArea(equal_center_groups[i][0]))
                if np.abs(distance-2*contourWidth) < contourWidth/2 or np.abs(distance-2*contourWidth*np.sqrt(2)) < contourWidth/2:  # most likely a locator box
                    distances[(i, j)] = distance   # distance between centerpoints[i] and centerpoints[j]

    if distances == {}:
        return equal_center_groups
    else:
        centers=equal_center_groups.keys()
        centersToDelete=[]
        delete=True
        for center in centers:
            for  (i, j) in distances.keys():
                if center == centers[i] or center == centers[j]:
                    delete=False
                    break
            if delete is True:
                centersToDelete.append(center)

    newCenters= [center for center in centers if center not in centersToDelete]
    newEqualCenterGroups={center:equal_center_groups[center] for center in newCenters}
    return newEqualCenterGroups

import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os

def loadImage(imgPath):
    """ Load Image from the given path, convert it to grayscale and return it """
    img = cv2.imread(imgPath) # 0 Because : We only need 1 color channel(?)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return thresh


def showImage(image, title:str="image",cmap=None)->None:
    if cmap is not None:
        plt.imshow(image,cmap=cmap)
    elif len(image.shape) == 3:    # means that the image is colored
        plt.imshow(image)
    elif len(image.shape) == 2:         # means that the image is grayscale
        plt.imshow(image,cmap='gray')    
    plt.title(title)
    plt.show()

def printPixelWindow(image, point , window_size=5):
    # Get the dimensions of the image
    x,y=point
    
    height, width = image.shape[:2]

    print(f"point: ({x},{y}) , value : {image[y, x]}")
    # Calculate the starting and ending indices for the window
    start_x = max(0, x - window_size // 2)
    start_y = max(0, y - window_size // 2)
    end_x = min(width - 1, x + window_size // 2)
    end_y = min(height - 1, y + window_size // 2)

    # Extract the window from the image
    for j in range(start_y, end_y + 1):
        for i in range(start_x, end_x + 1):
            if i == x and j == y:
                print(f" x ", end=" ")
            else:
                print(" 0 " if image[j, i]==0 else image[j, i], end=" ")
        print()  # Move to the next line for the next row of pixels

def drawCornerPoints(image,corners):
    # Filter out perfect 90-degree corners
    filtered_corners = []
    for corner in corners:
        x, y = corner.ravel()
        filtered_corners.append((x, y))

    # Visualize the detected corners on the original image
    for corner in filtered_corners:
        x, y = corner
        cv2.circle(image, (x, y), 10, (0, 255, 0), -1)

    return image


def drawContours(image, contours):
    """ 
    draw the outline of a contour contours on the bgr image
    """
    newImage=image.copy()
    for i in range(len(contours)):
        cv2.drawContours(newImage,contours,i,(0,255,0),4)
        i = i + 1
    return newImage

def drawContourPoints(image, contours,isPoints=False):
    # Iterate over each contour
    if type(contours) is not list and type(contours) is not tuple:
        if not isPoints:
            for contour in contours:
                for point in contour:
                    # Extract x and y coordinates of the point
                    x, y = point[0]
                    # Draw a point (circle) on the image at the current point
                    cv2.circle(image, (x, y), 10, (0, 0, 255), -1)  # Red color, filled circle
        else:
            for point in contours:
                # Extract x and y coordinates of the point
                x, y = point[0]
                # Draw a point (circle) on the image at the current point
                cv2.circle(image, (x, y), 10, (0, 0, 255), -1)  # Red color, filled circle
    else:
        if isPoints:
            for i,point in enumerate(contours):
                # Extract x and y coordinates of the point
                x, y = point[0]
                
                # Draw a point (circle) on the image at the current point
                cv2.circle(image, (x, y), 10, (0, 0, 255), -1)  # Red color, filled circle
        else:
            for j,contour in enumerate(contours):
                # Iterate over each point in the contour
                for i,point in enumerate(contour):
                    # Extract x and y coordinates of the point
                    x, y = point[0]
                    
                    # Draw a point (circle) on the image at the current point
                    cv2.circle(image, (x, y), 10, (0, 0, 255), -1)  # Red color, filled circle

    return image

def drawContourPixels(image, contours):
    # Iterate over each contour
    if type(contours) is not list and type(contours) is not tuple:
        for point in contours:
            # Extract x and y coordinates of the point
            x, y = point[0]
            # Draw a point (circle) on the image at the current point
            image[y, x] = (0, 0, 255)  # Red color, filled circle
    else:
        for contour in contours:
            # Iterate over each point in the contour
            for point in contour:
                # Extract x and y coordinates of the point
                x, y = point[0]
                # Draw a point (circle) on the image at the current point
                image[y, x] = (0, 0, 255)  # Red color, filled circle

    return image

def getImagePathsInDirectory(folder_path):
    paths=[]
    # Iterate over files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is an image (you can add more image extensions if needed)
        if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Read the image
            image_path = os.path.join(folder_path, file_name)
            paths.append(image_path)
    return paths


def orderPoints(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)

    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]  # topleft
    new_points[3] = points[np.argmax(add)] # bottomright

    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]     # topright
    new_points[2] = points[np.argmax(diff)]    # bottomleft

    return new_points


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area


def locatorContours2(image,maxCorners=500,minDistance=6,showImages=False, qualityLevel=0.25,invertColors=False):
    contours=getContours3(image=image,maxCorners=maxCorners,minDistance=minDistance,showImages=showImages,qualityLevel=qualityLevel, invertColors=invertColors)
    new_contours = []

    for i,contour in enumerate(contours):   # remove the contours that are not square 3 is accepted for error
        if len(contour) ==3 or len(contour) == 4:
            new_contours.append(contour)
            
    return new_contours

def getContours3(image,maxCorners=500,minDistance=6,showImages=False, qualityLevel=0.25, invertColors=False):
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
    
    contours=locatorContours(thresh)
    
    
    corners = cv2.goodFeaturesToTrack(thresh, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance)
    corners = np.int0(corners)
    
    corners=list(corners)
    originalCorners= corners.copy()
    new_contours = []  # List to store new contours
    for contour in contours:
    # List to store new contour points
        new_contour_points = []

        # Iterate over hull points
        for contourPoint in contour:
            x, y = contourPoint.ravel()

            # Check if hull point is close to any corner
            for i,corner in enumerate(corners):
                cx, cy = corner.ravel()
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)

                # If hull point is close to a corner, add it to new contour points
                if dist < 5:  # Adjust threshold as needed
                    new_contour_points.append(np.array([[cx,cy]]))  #[[x,y],[x,y]]
                    del corners[i]
                    break

        # new_contour_points = np.array(new_contour_points)  #[[[]]]  
        # Append new contour to list of new contours
        new_contours.append(new_contour_points)
    
    if showImages is True:
        _, axarr = plt.subplots(nrows=1, ncols=3, figsize=(10,5)) # figsize is in inches, yuck
        plt.sca(axarr[0]); plt.title('Contour Points'); plt.imshow(drawContourPoints(image.copy(),contours));
        plt.sca(axarr[1]); plt.title('corner Points'); plt.imshow(drawCornerPoints(image.copy(),originalCorners));
        plt.sca(axarr[2]); plt.title('new contour points'); plt.imshow(drawContourPoints(image.copy(),new_contours));
        plt.show()

    return new_contours


def getContours2(image,maxCorners=500,minDistance=6,showImages=False, qualityLevel=0.25, invertColors=False):
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
    
    contours=locatorContours(thresh)
    
    
    corners = cv2.goodFeaturesToTrack(thresh, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance)
    corners = np.int0(corners)
    
    
    new_contours = []  # List to store new contours
    for contour in contours:
    # List to store new contour points
        new_contour_points = []

        # Iterate over hull points
        for contourPoint in contour:
            x, y = contourPoint.ravel()

            # Check if hull point is close to any corner
            for corner in corners:
                cx, cy = corner.ravel()
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)

                # If hull point is close to a corner, add it to new contour points
                if dist < 10:  # Adjust threshold as needed
                    new_contour_points.append(contourPoint)

        # Convert new contour points to numpy array
        new_contour_points = np.array(new_contour_points)

        # Append new contour to list of new contours
        new_contours.append(new_contour_points)
    
    if showImages is True:
        _, axarr = plt.subplots(nrows=1, ncols=3, figsize=(10,5)) # figsize is in inches, yuck
        plt.sca(axarr[0]); plt.title('Contour Points'); plt.imshow(drawContourPoints(image.copy(),contours));
        plt.sca(axarr[1]); plt.title('corner Points'); plt.imshow(drawCornerPoints(image.copy(),corners));
        plt.sca(axarr[2]); plt.title('new contour points'); plt.imshow(drawContourPoints(image.copy(),new_contours));
        plt.show()

    return new_contours


def locatorContours(image):
    """ 
    image: Grayscale thresholded Image, or use loadImage() 
    
    returns the contours of the locator boxes | None
    """
    imageWidth, imageHeight =  image.shape[0:2]
    
    contours,heirarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    potential_boxes = ([])
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if abs(aspect_ratio - 1) > 0.1: # Aspect ratio should be close to 1 if it is a square-like or rectangular shape
            continue # Not what we are looking for.
        min_size = min(w, h)
        if min_size < 0.05 * imageWidth:  # Img_width [Should] be the same for all images.          
            continue                    
                                        # If the Rectangle is TOO small -> It is not the locator box.
                                        # Probably doesn't need to be a % of the image. ( Could just use some number )

        # Potential locator box based on heuristics
        potential_boxes.append(contour)

    return potential_boxes


def fixFlippedFixedQr(image,contours):
    """ 
    if Qr is normalized and flipped, this function will fix it.
    * return fixed image
    """
    imageWidth, imageHeight =  image.shape[0:2]

    fixedImage = image
    is_top_left = is_top_right = is_bottom_left = is_bottom_right = False
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)

        center_x, center_y = (x + x + w) // 2, (y + y + h) // 2
        image_center_x, image_center_y = imageWidth // 2, imageHeight // 2

        # Check for top-left, top-right, or bottom-left corner based on relative position
        if not is_top_left:   # if didnt find top left in the previous iterations continue checkign
            is_top_left =( center_x < (image_center_x * 0.4)) and (center_y < (image_center_y * 0.4))
        
        if not is_top_right:
            is_top_right =( center_x > (image_center_x * 1.6)) and (center_y < (image_center_y * 0.4))
        
        if not is_bottom_left:
            is_bottom_left =( center_x < (image_center_x * 0.4)) and (center_y > (image_center_y * 1.6))
        
        if not is_bottom_right:
            is_bottom_right =( center_x > (image_center_x * 1.6)) and (center_y > (image_center_y * 1.6))
        
        if is_top_left and is_top_right and is_bottom_left and is_bottom_right:
            print("4 locator boxes found.")
            return fixedImage

    # now check the flags of locator positions
    if is_top_right and not is_bottom_left and is_bottom_right and is_top_left:
        fixedImage = cv2.flip(image, 1)    # Flip the image around the y-axis

    if is_top_right and  is_bottom_left and is_bottom_right and not is_top_left:
        fixedImage = cv2.flip(image, -1)  # Flip the image around both the x-axis and y-axis

    if not is_top_right and is_bottom_left and is_bottom_right and is_top_left:
        fixedImage = cv2.flip(image, 0) # Flip the image around the x-axis        
    return fixedImage


def DetectPositionPattern(imgT):
    """ 
    imgT  : Grayscale thresholded Image
    """
    img_width  = 1012
    img_height = 1012

    contours,heirarchy = cv2.findContours(imgT,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    potential_boxes = ([])
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if abs(aspect_ratio - 1) > 0.2: # Aspect ratio should be close to 1 if it is a square-like or rectangular shape
            continue # Not what we are looking for.
        min_size = min(w, h)
        if min_size < 0.1 * img_width:  # Img_width [Should] be the same for all images.          
            continue                    
                                        # If the Rectangle is TOO small -> It is not the locator box.
                                        # Probably doesn't need to be a % of the image. ( Could just use some number )

        center_x, center_y = (x + x + w) // 2, (y + y + h) // 2
        image_center_x, image_center_y = img_width // 2, img_height // 2

        # Check for top-left, top-right, or bottom-left corner based on relative position
        is_top_left = center_x < image_center_x * 0.4 and center_y < image_center_y * 0.4
        is_top_right = center_x > image_center_x * 1.6 and center_y < image_center_y * 0.4
        is_bottom_left = center_x < image_center_x * 0.4 and center_y > image_center_y * 1.6


        if not (is_top_left or is_top_right or is_bottom_left): # Wrong Place.
            continue

        # Potential locator box based on heuristics
        potential_boxes.append(contour)
    return potential_boxes


def resize_image(image, target_width, target_height):
    # Calculate the aspect ratio of the image
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height

    # Calculate the new dimensions while maintaining aspect ratio
    if target_width is None:
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
    elif target_height is None:
        new_height = int(target_width / aspect_ratio)
        new_width = target_width
    else:
        new_width = target_width
        new_height = target_height

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def rotate_and_flip(image, rotateCode=0, flip_code=None):
    """ 
    rotateCode: This parameter specifies the direction of rotation and can take one of three values: 
                cv2.ROTATE_90_CLOCKWISE | cv2.ROTATE_180 | cv2.ROTATE_90_COUNTERCLOCKWISE
    flipCode:
            0: Flip vertically (around the x-axis).
            1: Flip horizontally (around the y-axis).
            -1: Flip both vertically and horizontally.
    """
    # Rotate the image by the specified angle (in degrees)
    if rotateCode == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotateCode == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif rotateCode == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Flip the image
    if flip_code is not None:
        image = cv2.flip(image, flip_code)

    return image


# Function to check if a point is a true corner
def isCornerTrue(image, point, radius=3, threshold=0.5):
    # Extract the coordinates of the point
    x, y = point[0]

    # Extract the pixel value at the point
    pixel_value = image[y, x]

    # Count the number of white pixels in the surrounding area
    white_count = 0
    for i in range(x - radius, x + radius + 1):
        for j in range(y - radius, y + radius + 1):
            if image[j, i] == 255:
                white_count += 1

    # Check if the last point is a true corner based on the criteria
    return pixel_value == 0 and white_count > (2 * radius + 1) ** 2 * threshold

def removeQZ(img):
    start_row = -1
    start_col = -1
    end_row = -1
    end_col = -1

    for row_index, row in enumerate(img):
        for pixel in row:
            if pixel != 255:
                start_row = row_index
                break
        if start_row != -1:
            break

    for row_index, row in enumerate(img[::-1]):
        for pixel in row:
            if pixel != 255:
                end_row = img.shape[0] - row_index
                break
        if end_row != -1:
            break

    for col_index, col in enumerate(cv2.transpose(img)):
        for pixel in col:
            if pixel != 255:
                start_col = col_index
                break
        if start_col != -1:
            break

    for col_index, col in enumerate(cv2.transpose(img)[::-1]):
        for pixel in col:
            if pixel != 255:
                end_col = img.shape[1] - col_index
                break
        if end_col != -1:
            break
    qr_no_quiet_zone = img[start_row:end_row, start_col:end_col]
    return qr_no_quiet_zone


def removeRedundantCornerPoints(contour, threshold=5):
    """ 
    Function that removes redundant corner points from a contour.
    
    Parameters
    ----------
    contour : numpy.ndarray
        The contour points to process.
        
    threshold : optional distance threshold between points. The default is 10.
    
    """
    new_contour = []
    for point in contour:
        # Add the first point or a point that is sufficiently far from the previous point
        if len(new_contour) == 0 or cv2.norm(point - new_contour[-1]) > threshold:
            new_contour.append(point)
    return np.array(new_contour)

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
    
    contours,heirarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    corners=list(corners)
    if showImages is True:
        originalCorners= corners.copy()
    new_contours = []  # List to store new contours
    flag=False
    for contour in contours:
    # List to store new contour points
        new_contour_points = []
        showImage(drawContourPoints(image.copy(),contour,True),title="contour")

        # Iterate over hull points
        for contourPoint in contour:
            if len(new_contour_points) >4:
                flag=True
                break

            x, y = contourPoint.ravel()
            # Check if hull point is close to any corner
            corners1=corners.copy()
            for i,corner in enumerate(corners):
                cx, cy = corner.ravel()
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)

                # If hull point is close to a corner, add it to new contour points
                if dist < 5:  # Adjust threshold as needed
                    new_contour_points.append(np.array([np.array([cx,cy])]))  #[[x,y],[x,y]]
                    del corners1[i]
                    break
            corners=corners1
        showImage(drawContourPoints(image.copy(),new_contour_points,True),title="new contour")
        if flag is True:    # if more than 4 corners are found, then the contour isn't a locator box
            continue
        new_contours.append(new_contour_points)
        

    new_contours = [contour for contour in new_contours if len(contour) > 2]  # Filter out contours with less than 3 points, which might be useless
    new_contours= np.array(new_contours)
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

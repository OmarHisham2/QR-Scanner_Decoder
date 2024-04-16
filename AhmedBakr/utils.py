import cv2 
import numpy as np
import matplotlib.pyplot as plt

def apply_linear_filter(padded_image, filter_size, filter):
    result = np.zeros((3,3))
    for i in range(0, padded_image.shape[0] - filter_size + 1):
        for j in range(0, padded_image.shape[1] - filter_size + 1):
            window = padded_image[i:i+filter_size, j:j+filter_size]
            result[i, j] = np.sum(filter * window)/np.sum(filter)
    return result

def apply_median_filter(padded_image):
    result = np.zeros((3,3))
    for i in range(0, padded_image.shape[0] - filter_size + 1):
        for j in range(0, padded_image.shape[1] - filter_size + 1):
            window = padded_image[i:i+filter_size, j:j+filter_size]
            result[i, j] = np.median(window)
    return result

def showImage(image, title:str="image",cmap=None)->None:
    if cmap is not None:
        plt.imshow(image,cmap=cmap)
    elif len(image.shape) == 3:    # means that the image is colored
        plt.imshow(image)
    elif len(image.shape) == 2:         # means that the image is grayscale
        plt.imshow(image,cmap='gray')    
    plt.title(title)
    plt.show()

def loadImage(imgPath):
    """ Load Image from the given path, convert it to grayscale and return it """
    img = cv2.imread(imgPath) # 0 Because : We only need 1 color channel(?)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return thresh

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

def locatorContours(image):
    """ 
    image: Grayscale thresholded Image, or use loadImage() 
    
    returns the contours of the locator boxes | None
    """
    img_width  = 1012
    img_height = 1012
    contours,heirarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    potential_boxes = ([])
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if abs(aspect_ratio - 1) > 0.1: # Aspect ratio should be close to 1 if it is a square-like or rectangular shape
            continue # Not what we are looking for.
        min_size = min(w, h)
        if min_size < 0.2 * img_width:  # Img_width [Should] be the same for all images.          
            continue                    
                                        # If the Rectangle is TOO small -> It is not the locator box.
                                        # Probably doesn't need to be a % of the image. ( Could just use some number )

        # Potential locator box based on heuristics
        potential_boxes.append(contour)

    if len(potential_boxes) == 9:
        print('Passed!')
        return potential_boxes
    print('Failed.')
    print('Potential Boxes Found : '+ str(len(potential_boxes)))
    return potential_boxes

def fixFlippedFixedQr(image,contours):
    """ 
    if Qr is normalized and flipped, this function will fix it.
    * return fixed image
    """
    img_width  = 1012
    img_height = 1012
    fixedImage = image
    is_top_left = is_top_right = is_bottom_left = is_bottom_right = False
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)

        center_x, center_y = (x + x + w) // 2, (y + y + h) // 2
        image_center_x, image_center_y = img_width // 2, img_height // 2

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


def drawContours(image, contours):
    """ 
    draw the  contours on the bgr image
    """
    newImage=image.copy()
    for i in range(len(contours)):
        cv2.drawContours(newImage,contours,i,(0,255,0),4)
        i = i + 1
    return newImage

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
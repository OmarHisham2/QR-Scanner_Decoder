def loadImageT(imgPath):
    img = cv2.imread(imgPath) # 0 Because : We only need 1 color channel(?)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if img.min() > 88:
        _, thresh = cv2.threshold(img, img.mean(), 255, cv2.THRESH_BINARY)
    else:
        _, thresh = cv2.threshold(img, 88, 255, cv2.THRESH_BINARY)
    return thresh



def loadImageRGB(imgPath): # This is used if we want to visualize something colored in our Plotting.
    img = cv2.imread(imgPath) 
    return img


## CONTOUR RELATED FUNCTIONS ##


def findContours(imgT): # This is used to return the contours of an image.
    contours,heirarchy = cv2.findContours(imgT,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def drawContours(imgRGB,contours): # For Visualization Purposes Only - Used to draw the contours on an image.
    for i in range(len(contours)):
        cv2.drawContours(imgRGB,contours,i,(0,255,0),4)
        i = i + 1
    plt.imshow(imgRGB,cmap='gray')

def filterContours(contours):
    img_width  = 1012
    img_height = 1012
    potential_boxes = ([])
    total_boxes = ([])
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if abs(aspect_ratio - 1) > 0.04: # Aspect ratio should be close to 1 if it is a square-like or rectangular shape
            continue # Not what we are looking for.
        min_size = min(w, h)
        max_size = max(w, h)
        if (min_size < 0.13 * img_width) or (max_size > 0.4 * img_width):  # Img_width [Should] be the same for all images.          
            continue                    
                                        # If the Rectangle is TOO small -> It is not the locator box.
                                        # Probably doesn't need to be a % of the image. ( Could just use some number )

        center_x, center_y = (x + x + w) // 2, (y + y + h) // 2
        image_center_x, image_center_y = img_width // 2, img_height // 2

        # Check for top-left, top-right, or bottom-left corner based on relative position
        is_top_left = center_x < image_center_x * 0.4 and center_y < image_center_y * 0.4
        is_top_right = center_x > image_center_x * 1.6 and center_y < image_center_y * 0.4
        is_bottom_left = center_x < image_center_x * 0.4 and center_y > image_center_y * 1.6
        is_bottom_right = ( center_x > (image_center_x * 1.6)) and (center_y > (image_center_y * 1.6))



        if not (is_top_left or is_top_right or is_bottom_left): # Wrong Place.
            if is_bottom_right:
                total_boxes.append(contour)
            continue

        # Potential locator box based on heuristics
        total_boxes.append(contour)
        potential_boxes.append(contour)

    if ( len(potential_boxes) == 9 ):
        return (True,0)
    elif ( len(total_boxes) == 9):
        return (False,1)
    else:
        return(False,2)

def filterContoursV2(contours): # Used For Debugging.
    img_width  = 1012
    img_height = 1012
    potential_boxes = ([])
    total_boxes = ([])
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if abs(aspect_ratio - 1) > 0.04: # Aspect ratio should be close to 1 if it is a square-like or rectangular shape
            continue # Not what we are looking for.
        min_size = min(w, h)
        max_size = max(w, h)
        if (min_size < 0.13 * img_width) or (max_size > 0.4 * img_width):  # Img_width [Should] be the same for all images.          
            continue                    
                                        # If the Rectangle is TOO small -> It is not the locator box.
                                        # Probably doesn't need to be a % of the image. ( Could just use some number )

        center_x, center_y = (x + x + w) // 2, (y + y + h) // 2
        image_center_x, image_center_y = img_width // 2, img_height // 2

        # Check for top-left, top-right, or bottom-left corner based on relative position
        is_top_left = center_x < image_center_x * 0.4 and center_y < image_center_y * 0.4
        is_top_right = center_x > image_center_x * 1.6 and center_y < image_center_y * 0.4
        is_bottom_left = center_x < image_center_x * 0.4 and center_y > image_center_y * 1.6
        is_bottom_right = ( center_x > (image_center_x * 1.6)) and (center_y > (image_center_y * 1.6))



        if not (is_top_left or is_top_right or is_bottom_left): # Wrong Place.
            if is_bottom_right:
                total_boxes.append(contour)
            continue

        # Potential locator box based on heuristics
        total_boxes.append(contour)
        potential_boxes.append(contour)

    if ( len(potential_boxes) == 9 ):
        return (True,0,potential_boxes)
    elif ( len(total_boxes) == 9):
        return (False,1,total_boxes)
    else:
        return(False,2,total_boxes)

## CONTOUR RELATED FUNCTIONS ##

## Draw Locator Boxes ##

img = cv2.imread('TC/01.png')

imgT = loadImageT('TC/01.png')


contours = findContours(imgT)

_,_,filteredcnts = filterContoursV2(contours)

whitemask = np.ones((1012,1012),dtype=np.uint8)


"""Bottom Left Drawing"""
x1, y1, w1, h1 = cv2.boundingRect(filteredcnts[0])
x2, y2, w2, h2 = cv2.boundingRect(filteredcnts[1])

min_x = min(x1, x2)  # Leftmost X
max_x = max(x1 + w1, x2 + w2)  # Rightmost X
min_y = min(y1, y2)  # Topmost Y
max_y = max(y1 + h1, y2 + h2)  # Bottommost Y



for y in range(min_y, max_y):
    for x in range(min_x, max_x):
        whitemask[y, x] = 0  # Set pixel value to 0

x1, y1, w1, h1 = cv2.boundingRect(filteredcnts[1])
x2, y2, w2, h2 = cv2.boundingRect(filteredcnts[2])

min_x = min(x1, x2)  # Leftmost X
max_x = max(x1 + w1, x2 + w2)  # Rightmost X
min_y = min(y1, y2)  # Topmost Y 
max_y = max(y1 + h1, y2 + h2)  # Bottommost Y



for y in range(min_y, max_y):
    for x in range(min_x, max_x):
        whitemask[y, x] = 255  

x, y, w, h = cv2.boundingRect(filteredcnts[2]) 
cv2.rectangle(whitemask, (x, y), (x + w, y + h), color=(0,0,0), thickness=-1) 

"""Bottom Left Done"""


"""Top Left Drawing"""
x1, y1, w1, h1 = cv2.boundingRect(filteredcnts[3])
x2, y2, w2, h2 = cv2.boundingRect(filteredcnts[4])

min_x = min(x1, x2)  # Leftmost X
max_x = max(x1 + w1, x2 + w2)  # Rightmost X
min_y = min(y1, y2)  # Topmost Y 
max_y = max(y1 + h1, y2 + h2)  # Bottommost Y



for y in range(min_y, max_y):
    for x in range(min_x, max_x):
        whitemask[y, x] = 0  # Set pixel value to 0

x1, y1, w1, h1 = cv2.boundingRect(filteredcnts[4])
x2, y2, w2, h2 = cv2.boundingRect(filteredcnts[5])

min_x = min(x1, x2)  # Leftmost X
max_x = max(x1 + w1, x2 + w2)  # Rightmost X
min_y = min(y1, y2)  # Topmost Y 
max_y = max(y1 + h1, y2 + h2)  # Bottommost Y



for y in range(min_y, max_y):
    for x in range(min_x, max_x):
        whitemask[y, x] = 255  

x, y, w, h = cv2.boundingRect(filteredcnts[5]) 
cv2.rectangle(whitemask, (x, y), (x + w, y + h), color=(0,0,0), thickness=-1) 

"""Top Right Done"""


"""Top Left Drawing"""
x1, y1, w1, h1 = cv2.boundingRect(filteredcnts[6])
x2, y2, w2, h2 = cv2.boundingRect(filteredcnts[7])

min_x = min(x1, x2)  # Leftmost X
max_x = max(x1 + w1, x2 + w2)  # Rightmost X
min_y = min(y1, y2)  # Topmost Y 
max_y = max(y1 + h1, y2 + h2)  # Bottommost Y



for y in range(min_y, max_y):
    for x in range(min_x, max_x):
        whitemask[y, x] = 0  # Set pixel value to 0

x1, y1, w1, h1 = cv2.boundingRect(filteredcnts[7])
x2, y2, w2, h2 = cv2.boundingRect(filteredcnts[8])

min_x = min(x1, x2)  # Leftmost X
max_x = max(x1 + w1, x2 + w2)  # Rightmost X
min_y = min(y1, y2)  # Topmost Y 
max_y = max(y1 + h1, y2 + h2)  # Bottommost Y



for y in range(min_y, max_y):
    for x in range(min_x, max_x):
        whitemask[y, x] = 255  

x, y, w, h = cv2.boundingRect(filteredcnts[8]) 
cv2.rectangle(whitemask, (x, y), (x + w, y + h), color=(0,0,0), thickness=-1) 

def drawLocatorBoxes(img):
    return cv2.bitwise_and(img,img,mask=whitemask)


## Draw Locator Boxes ##



## CHECKING FUNCTIONS ##


def isAlmostInvisible(img): # Is Image Too bright?
    if (np.mean(img) > 195):
        return True
    return False

def isAlmostInvisibleDark(img): # Is Image Too Dark?
    if (np.mean(img) < 40 ):
        return True
    return False



def isSkewed(img):
    try:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    finally:
        if img.min() > 88:
            _, imageT = cv2.threshold(img, img.min(), 255, cv2.THRESH_BINARY)
        else:
            _, imageT = cv2.threshold(img, 88, 255, cv2.THRESH_BINARY)

        cnts = findContours(imageT)

        __,__,filteredCNTS = filterContoursV2(cnts)

        minTheta = 0

        for cnt in filteredCNTS:
            (x,y),(width,height),theta = cv2.minAreaRect(cnt)
            theta = theta - 90
            if(theta <= minTheta):
                minTheta = theta
        if (minTheta <= -3 and minTheta != -90):
            return (True,minTheta) # This Needs To Be Unskewed
        return (False,minTheta)



## CHECKING FUNCTIONS ##



## FIXING FUNCTIONS 


def rotateImg(img): 
    rotated = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    return rotated

def invertImg(img):
    return cv2.bitwise_not(img) # Or 255 - img

def threshHoldMean(img):
    _, thresh = cv2.threshold(img, img.mean(), 255, cv2.ADAPTIVE_THRESH_MEAN_C)
    return thresh


def fixSkew(img,angle):

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

## PRESPECTIVE RELATED ##

def biggestContours(contours,numberOfContours=1):
    # Sort the contours by area in descending order
    contours = [np.array(contour) for contour in contours]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:numberOfContours]
    return contours

def computeCenter(contours):
    # Combine contours into one array
    if len(contours)==0:
        return None
    
    combined_contour = np.vstack(contours)

    # Compute bounding box for the combined contours
    x, y, w, h = cv2.boundingRect(combined_contour)

    # Calculate center point of the bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    return (center_x, center_y)

def find_missing_point(point1, point2, known_point):  
    point1 = list(point1.ravel())
    point2 = list(point2.ravel())
    known_point = list(known_point.ravel())
    # Determine the direction vector of the first line
    direction_vector1 = np.array(point2) - np.array(point1)
    
    # Find the equation of the first line (y = mx + c)
    m1 = direction_vector1[1] / direction_vector1[0]  # Slope
    c1 = point1[1] - m1 * point1[0]  # Intercept
    
    
    # Find the equation of the second line
    m2 = m1  # Lines are parallel, so slopes are equal
    c2 = known_point[1] - m2 * known_point[0]  # Intercept

    # Calculate the length of the vector
    vector_length = np.linalg.norm(direction_vector1)
    
    # Find the x-coordinate of the intersection point
    intersection_x = known_point[0] + vector_length*np.cos(np.arctan(m1))
    
    # Find the y-coordinate of the intersection point
    intersection_y = known_point[1] + vector_length*np.sin(np.arctan(m1))
    
    return intersection_x, intersection_y


def AdjustPrespective(image):

    contours= ut.locatorContours2(image=image,invertColors=False,showImages=False)

    locatorBoxes=biggestContours(contours,numberOfContours=3)

    centerPoint=computeCenter(locatorBoxes)
    cv2.circle(center:=image.copy(), centerPoint, 10, (0, 255, 0), -1)


    for i,locatorbox in enumerate(locatorBoxes):  # order all points in each box
        locatorBoxes[i]=ut.orderPoints(locatorbox)

    topLeft= topRight= bottomRight= bottomLeft = 0  # corner points for the whole qr code
    for locator in locatorBoxes:
        x, y, w, h = cv2.boundingRect(locator)

        # Calculate center point of the bounding box
        center_x,center_y = x + w // 2,y + h // 2
        
        if center_x> centerPoint[0] and center_y< centerPoint[1]:
            topRight=locator[1]
        
        # elif center_x> centerPoint[0] and center_y> centerPoint[1]:
            # bottomRight=locator[3]
        
        elif center_x< centerPoint[0] and center_y< centerPoint[1]:
            topLeft=locator[0]
        
        elif center_x< centerPoint[0] and center_y> centerPoint[1]:
            bottomLeft=locator[2]

    cv2.circle(center, topRight.ravel(), 10, (0, 0, 255), -1)
    cv2.circle(center, bottomLeft.ravel(), 10, (0, 0, 255), -1)
    cv2.circle(center, topLeft.ravel(), 10, (0, 0, 255), -1)
    # cv2.circle(center, bottomRight.ravel(), 10, (0, 0, 255), -1)
    bottomRight=find_missing_point(topLeft,topRight,bottomLeft)
    bottomRight=(int(round(bottomRight[0])),int(round(bottomRight[1])))
    bottomRight=np.array([bottomRight])
    cv2.circle(center, bottomRight.ravel(), 10, (0, 0, 255), -1)

    width,height = 1012,1012


    pts1 = np.float32([topLeft.ravel(),topRight.ravel(),bottomLeft.ravel(),bottomRight.ravel()])
    pts2 = np.float32([[0,0],[0,width],[height,0],[width,height]])



    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    output = cv2.warpPerspective(image,matrix,(width,height))

    return output
## PRESPECTIVE RELATED ##


## FIXING FUNCTIONS 

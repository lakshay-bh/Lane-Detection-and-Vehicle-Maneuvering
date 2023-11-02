import cv2
import numpy as np

def canny(lane_image):
   #convert pic to grayscale
   gray = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY) 
   #blurring image(smoothening) using a 5x5 matrix and weighted average of pixels
   blur = cv2.GaussianBlur(gray,(5,5),0)
   #finding edges(large gradient/drastic change in intensity)
   edge = cv2.Canny(blur,50,150)
   return edge

def area_of_interest(lane_image):
    height = lane_image.shape[0]        
    width = lane_image.shape[1]
    area = np.array([[(-30,height-100),(width-75,height-100),(650,300)]])   #marking region of interest using x,y coordinates
    mask = np.zeros_like(lane_image)         #reference black image
    cv2.fillPoly(mask,area,255)               #area of interest super-imposed in black image
    masked = cv2.bitwise_and(lane_image,mask) #takes bitwise & of each bit of pixel to retain area of interest in canned image
    return masked

def coordinates(lane_image,parameters):
    slope,intercept = parameters
    y1 = lane_image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope) 
    return np.array([x1,y1,x2,y2])

def average_slope(lane_image,lines):     #taking average of slope interecpt to obtain single line
    left_fit = []        #coordinates of line on left
    right_fit = []       #coordinated of line on right
    points = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)   
        parameters = np.polyfit((x1,x2),(y1,y2),1)    #finds slope and y intercept of each line
        slope = parameters[0]
        intercept = parameters[1]
        if slope<0:       #categorizing lines as left and right
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit,axis=0)   #taking average of all lines on lhs to obtain a single line
    right_fit_average = np.average(right_fit,axis=0)
    left_line = coordinates(lane_image,left_fit_average)     #obtaining coordinates for 2 lines
    right_line = coordinates(lane_image,right_fit_average)
    # return np.array([left_line,right_line]),points
    return np.array([left_line,right_line])

def reference(frame,lines):
    left,right = average_slope(frame,lines)
    x1,y1,x3,y3 = left.reshape(4)
    x2,y2,x4,y4 = right.reshape(4)
    mid_x,mid_y = (x3+x4)/2,(y3+y4)/2
    limit = mid_x - x3
    return mid_x,mid_y,limit

def display(lane_image,lines):
    line_image = np.zeros_like(lane_image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
             cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)      #displays line in image with given color(rgb) and thickness as arguments
    return line_image

def marker(lane_image):
    mark = cv2.legacy.TrackerMedianFlow_create()
    box = cv2.selectROI(lane_image,False)
    mark.init(lane_image,box)
    return mark,box

def command(lane_image,dis,limit):
    if(dis>limit):
        if (dis>=-575 and dis<=575):
            cv2.putText(final_image, "CONTINUE STRAIGHT", (100,80), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        elif (dis>575):
            cv2.putText(final_image, "SLIGHT RIGHT", (100,80), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        else:
            cv2.putText(final_image, "SLIGHT LEFT", (100,80), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    else:
        cv2.putText(final_image, "OUT OF LANE", (100,80), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        
capture = cv2.VideoCapture("test_video.mp4")       #input video
_,image = capture.read()  
lane_image = np.copy(image)

# edge = canny(lane_image)
# cropped = area_of_interest(edge)

# lines = cv2.HoughLinesP(cropped,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)  #finding best fit line for given points using hough transform(rho,theta and each bin)
# #arg 2 & 3 define size of each bin(rho,theta); arg 4 defines resolution or no. of intersection per bin to detect a line;arg 5 is a placeholder
# #arg 6 defines min length of line in pixel to be detected; arg 7 defines min gap in pixel between segment to be considered as a single line

# optimised = average_slope(lane_image,lines)

# lines_image = display(lane_image,optimised)

# final_image = cv2.addWeighted(lane_image,0.8,lines_image,1,1)

# display pic
# plt.imshow(edge)
# cv2.imshow('result',final_image)
# cv2.waitKey(0)

mark = cv2.legacy.TrackerMedianFlow_create()
box = cv2.selectROI(lane_image,False)
mark.init(lane_image,box)
# mark,box = marker(lane_image)

while (capture.isOpened()):
    _,frame = capture.read()                 #capture video frame by frame
    edge = canny(frame)
    cropped = area_of_interest(edge)

    lines = cv2.HoughLinesP(cropped,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)  #finding best fit line for given points using hough transform(rho,theta and each bin)
    #arg 2 & 3 define size of each bin(rho,theta); arg 4 defines resolution or no. of intersection per bin to detect a line;arg 5 is a placeholder
    #arg 6 defines min length of line in pixel to be detected; arg 7 defines min gap in pixel between segment to be considered as a single line

    optimised = average_slope(frame,lines)
    
    x,y,limit = reference(frame,lines)
    ref = (int(x),int(y))
    obj = ref[0]
    
    lines_image = display(frame,optimised)

    final_image = cv2.addWeighted(frame,0.8,lines_image,1,1)
    
    x_top,w = box[0],box[2]
    pos = x_top + (w/2);
    dis = obj-pos
    
    command(final_image,dis,limit)
    _,box = mark.update(final_image)

    #display pic
    #plt.imshow(edge)
    #cv2.fillPoly(final_image,np.int32([points]),color = [0,255,0])
    image = cv2.circle(final_image,ref,radius=3,color=(0,0,255),thickness=-1)
    cv2.imshow('result',final_image)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('p'):
            break
capture.release()
cv2.destroyAllWindows()
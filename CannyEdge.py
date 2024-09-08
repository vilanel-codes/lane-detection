import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    print(image.shape)
    y1 = image.shape[0]
    y2= int(y1*(7/9))
    x1= int((y1 - intercept)/slope)
    x2= int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters  = np.polyfit((x1,x2),(y1,y2), 1)
        print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope<0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope,intercept))
        print(left_fit)
        print(right_fit)
    left_fit_average=np.average(left_fit,axis=0)
    right_fit_average= np.average(right_fit,axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # blur = cv2.GaussianBlur(gray, (5,5),0)
    # canny=cv2.Canny(blur,50,150)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    median_intensity = np.median(blur)
    lower = int(max(0, 0.7 * median_intensity))
    upper = int(min(255, 1.3 * median_intensity))
    canny = cv2.Canny(blur, lower, upper)
    return canny

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0), 10)
    return line_image
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200, height),(1100,height),(550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image= cv2.bitwise_and(image, mask)
    return masked_image

def fit_polynomial(image, lines):
    left_x, left_y, right_x, right_y = [], [], [], []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        slope = (y2 - y1) / (x2 - x1)
        if slope < 0:
            left_x.extend([x1, x2])
            left_y.extend([y1, y2])
        else:
            right_x.extend([x1, x2])
            right_y.extend([y1, y2])
    
    if not left_x or not right_x:
        return None
    
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    
    plot_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]
    
    left_line = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
    right_line = np.array([np.transpose(np.vstack([right_fit_x, plot_y]))])
    return np.array([left_line, right_line])

cap = cv2.VideoCapture("Easy_Test.mp4")
while(cap.isOpened()):
    _, frame=cap.read()
    canny_image= canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines= cv2.HoughLinesP(cropped_image,2,np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # Fit polynomial if lines are detected
    # averaged_lines = fit_polynomial(frame, lines) if lines is not None else None
    averaged_lines= average_slope_intercept(frame, lines)
    line_image=display_lines(frame, averaged_lines)
    combo_image= cv2.addWeighted(frame, 0.8, line_image, 1,1)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

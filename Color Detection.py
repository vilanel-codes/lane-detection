import cv2
import numpy as np

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[0]
    polygons = np.array([
    [(248,1068),(1914,945),(931,308)]
    ])

    mask= np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)

    masked_image= cv2.bitwise_and(image, mask)
    return masked_image

cap = cv2.VideoCapture("Bonus Test.mp4")
while(cap.isOpened()):
    _, image=cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define blue color range
    light_gray = np.array([0,0,0])
    dark_gray = np.array([255,17,255])
    #define kernel size
    kernel = np.ones((7,7),np.uint8)

    # Threshold the HSV image to get only gray colors
    mask = cv2.inRange(hsv, light_gray, dark_gray)
    # Remove unnecessary noise from mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask= region_of_interest(mask)

    # Bitwise-AND mask and original image
    color_detect = cv2.bitwise_and(image,image, mask= mask)
    color_detect[mask>0]=(0,255,0)
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.drawContours(color_detect, contours, -1, (0, 255, ), 3)

    combo_image= cv2.addWeighted(image, 0.8, output, 1,1)
    cv2.imshow("Color Detected",combo_image )
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

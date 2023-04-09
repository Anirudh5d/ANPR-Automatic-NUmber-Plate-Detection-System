import numpy as np
import imutils
import csv
import cv2
from pytesseract import pytesseract


pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

cap = cv2.VideoCapture('D:\TY Proj\Project\cars.mp4')


while(cap.isOpened()):

    ret, frame = cap.read()


    if ret:

        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 11)
        edges = cv2.Canny(bfilter, 200, 300)
        keypoints = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]


        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)

            if len(approx) == 4:
                location = approx
                break
            else:
                continue
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)


        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y)-20)
        (x2, y2) = (np.max(x), np.max(y)-10)
        cropped_img = gray[x1:x2 + 20, y1:y2 + 20]
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        wrd = pytesseract.image_to_string(cropped_img, lang='eng')
        print("wrd:", wrd)
        if wrd > 'MH 00 AA 0000':
            if len(wrd) >= 4:
                print("number plate is:", wrd)
                csvfile = open('D:\plates.csv', 'w')
                obj = csv.writer(csvfile)
                obj.writerow(wrd)
                csvfile.close()
                break
    cv2.imwrite('frame'+'.jpg', frame)
    cv2.imshow('frame', frame)
    cv2.waitKey(5)
print("the number plate:", wrd)
cap.release()
cv2.destroyAllWindows()

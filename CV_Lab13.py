from ultralytics import YOLO
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import time
import pandas as pda

seg_model = YOLO('yolov8n.pt')

#Task 4
cap = cv.VideoCapture(0)

width  = cap.get(cv.CAP_PROP_FRAME_WIDTH)   # float `width`
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float `height`
writeto = cv.VideoWriter(r"C:\Users\Taimoor\out_cameras.mp4",  cv.CAP_FFMPEG, cv.VideoWriter.fourcc('M','P','4','V'), 10, np.uint32((width, height)))
i = 0

print("started")
while (True):
    ret, img = cap.read()
    results = seg_model.predict(img)
    i = i + 1
    
    plot = results[0].plot()
    writeto.write(plot)

    cv.namedWindow('camera', cv.WINDOW_AUTOSIZE)
    cv.imshow('camera',plot)

    key = cv.waitKey(20)
    if key == 27: # exit on ESC
        break

cv.destroyAllWindows()
writeto.set(cv.CAP_PROP_FPS, int(i/10))
writeto.release()
cap.release()

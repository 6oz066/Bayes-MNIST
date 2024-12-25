import time
import cv2
import GS_cal as gs
from tqdm import tqdm

# Set the initial value
N=20
T=5

# Capturing the video cap and dealing
cv2.destroyAllWindows()
input_video = cv2.VideoCapture('sample-mp4-file.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 videos
print(fourcc)
output_video = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))

cnt = 0 # Counter for reading the first cap
for i in tqdm(range(22),'process'):
    ret, frame = input_video.read()
    if ret == True:
        # If it is the first cap
        if cnt == 0:
            cv2.imshow('frame',frame)
            new_frame,msevalue,params= gs.picture_deal(frame,N, T)
            output_video.write(new_frame)
            cnt += 1
            gs.showmse(msevalue)
        # For the other cap
        else:
            new_frame,msevalue,params= gs.picture_deal_oo(frame,N, T,params)
            output_video.write(new_frame)
            # cv2.imshow('frame',new_frame)

# Release the resource
input_video.release()
output_video.release()
cv2.destroyAllWindows()
import cv2 

cap = cv2.VideoCapture('D://訓練影片//GO//FCUAI_rppgdata//ex3//2.mp4')

fps = cap.get(5)
frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

print(fps)
print(frame)
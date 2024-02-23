import cv2
from threading import Thread
import time
from pyVHR.extraction.utils import *
from datetime import datetime
from params import Params


class VideoCapture:
    # bufferless VideoCapture
    def __init__(self, name, sharedData, fps=None, sleep=False, resize=False):
        self.cap = cv2.VideoCapture(name)               # ,cv2.CAP_DSHOW  加進去會強制使用相機
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('I', '4', '2', '0'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 30)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.sleep = sleep
        self.resize = resize
        self.fps = None
        if fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.fps = fps
        self.sd = sharedData
        self.t = Thread(target=self._reader)
        self.t.daemon = True  # when process ends all deamon will be killed
        self.t.start()

    # read frames as soon as they are available
    def _reader(self):
        i=0
        start_time = time.perf_counter()
        while True:
            if not self.sd.q_stop_cap.empty():  # GUI stopped
                self.sd.q_stop_cap.get()
                self.sd.q_stop.put(0)  # stop VHR model
                self.sd.q_frames.put(0)
                break
            ret, frame = self.cap.read()
            if not ret:  # IO error
                self.sd.q_stop.put(0)  # stop VHR model
                self.sd.q_frames.put(0)  # end extraction -> send number
                break
            if self.resize:
                h, w = frame.shape[0], frame.shape[1]
                if h != 480 or w != 640:
                    frame = cv2.resize(frame, (640, int(640*h/w)),
                                       interpolation=cv2.INTER_NEAREST)
                    
            if Params.videoFileName != 0:               #不使用鏡頭時
                
                time.sleep(0.0333)
                i+=1
                self.sd.q_frames.put(frame)

                localtime = datetime.now()
                current = localtime.strftime("%Y-%m-%d %H:%M:%S.%f")
                if Params.startTime is None:
                    Params.startTime = current
                    print("start: {}".format(Params.startTime))

                self.sd.q_times.put(current)
            else:
                i+=1
                self.sd.q_frames.put(frame)

                localtime = datetime.now()
                current = localtime.strftime("%Y-%m-%d %H:%M:%S.%f")
                if Params.startTime is None:
                    Params.startTime = current
                    print("start: {}".format(Params.startTime))

                self.sd.q_times.put(current)
        self.cap.release()


"""
if __name__ == "__main__":
    import cv2
    from pyVHR.realtime.VHRroutine import SharedData
    sd = SharedData()
    cap = VideoCapture(
        "/home/frea/Documents/VHR/LGI_PPGI/lgi_alex/alex_resting/cv_camera_sensor_stream_handler.avi", fps=25, sharedData=sd)

    # Check if the webcam is opened correctly
    for i in range(10):
        frame = sd.q_frames.get()
        #cv2.imshow('Input', frame)
        print(frame.shape)
    sd.q_stop_cap.put(0)
    time.sleep(2)
"""

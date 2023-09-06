import sys
sys.path.append("D://rPPG//FCUAI_rPPG_workplace//pyVHR")
import warnings
warnings.filterwarnings('ignore')
from time import sleep
from video_capture import VideoCapture
from params import Params
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from importlib import import_module, util
from pyVHR.datasets.dataset import datasetFactory
from pyVHR.utils.errors import getErrors, printErrors, displayErrors
from extraction.sig_processing import *
from extraction.sig_extraction_methods import *
from extraction.skin_extraction_methods import *
from BVP.BVP import *
from BPM.BPM import *
from BVP.methods import *
from BVP.filters import *
from parallel import *
import tensorflow as tf
import PySimpleGUI as sg
import numpy as np
import plotly.graph_objects as go
import queue
from datetime import datetime
import threading


class SharedData:
    def __init__(self):
        self.q_bpm = queue.Queue()
        self.q_video_image = queue.Queue()
        self.q_skin_image = queue.Queue()
        self.q_patches_image = queue.Queue()
        self.q_stop = queue.Queue()
        self.q_stop_cap = queue.Queue()
        self.q_frames = queue.Queue()
        ###TEST###
        self.q_times = queue.Queue()
        self.q_saveTimes = queue.Queue()
        self.q_bpm2 = queue.Queue()
        self.q = queue.Queue()
        self.q2 = queue.Queue()
        self.q_first_face_ldmks = queue.Queue()
        self.q_second_face_ldmks = queue.Queue()
        self.q_face_range = queue.Queue()
        self.q_face_range2 = queue.Queue()
        ###TEST###


def VHRroutine(sharedData):

    ###TEST###
    global bvpSave
    global bpmSave
    global bpmTime 
    bvpSave = []
    bpmSave = []
    bpmTime = []

    global loc5
    loc5 = []
    ctr = 0
    currentBPM = 0
    currentBPM2 = 0
    foursecbpm = []
    state = 0
    state2 = 0
    interval = 3                #幾秒算一次
    patchNum = 0
    skinlen = 0
    movement = 5
    flag = 0                    #作為跳過第30偵的開關，若為0則關。第29偵會開啟
    global last_bpm
    global bpm_num

    st1 = "The central point of face:"
    st2 = "Movement:"
    st3 = "Moving!"

    fpsCount = 0
    passTime = 0
        ###TEST###
    min_x  = 1000
    min_y  = 1000
    max_x = 0
    max_y = 0
    
    min_x2  = 1000
    min_y2  = 1000
    max_x2 = 0
    max_y2 = 0
    ###TEST###
    ###TEST###

    sig_ext_met = None
    ldmks_regions = None
    # Holistic settings #
    if Params.approach == 'holistic':
        sig_ext_met = holistic_mean
    # Patches settings #
    elif Params.approach == 'patches':
        # extraction method
        if Params.type == "mean" and Params.patches == "squares":       #預設走這裡
            sig_ext_met = landmarks_mean
        elif Params.type == "mean" and Params.patches == "rects":
            sig_ext_met = landmarks_mean_custom_rect
        elif Params.type == "median" and Params.patches == "squares":
            sig_ext_met = landmarks_median
        elif Params.type == "median" and Params.patches == "rects":
            sig_ext_met = landmarks_median_custom_rect
        # patches dims
        if Params.patches == "squares":
            ldmks_regions = np.float32(Params.squares_dim)
        elif Params.patches == "rects":
            ldmks_regions = np.float32(Params.rects_dims)

    SignalProcessingParams.RGB_LOW_TH = np.int32(
        Params.sig_color_low_threshold)
    SignalProcessingParams.RGB_HIGH_TH = np.int32(
        Params.sig_color_high_threshold)
    SkinProcessingParams.RGB_LOW_TH = np.int32(
        Params.skin_color_low_threshold)
    SkinProcessingParams.RGB_HIGH_TH = np.int32(
        Params.skin_color_high_threshold)

    color = np.array([Params.font_color[0],
                      Params.font_color[1], 
                      Params.font_color[2]], dtype=np.uint8)

    skin_ex = None
    target_device = 'GPU' if Params.cuda else 'CPU'
    if Params.skin_extractor == 'convexhull':
        skin_ex = SkinExtractionConvexHull(target_device)
    elif Params.skin_extractor == 'faceparsing':
        skin_ex = SkinExtractionFaceParsing(target_device)

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    PRESENCE_THRESHOLD = 0.5
    VISIBILITY_THRESHOLD = 0.5

    if Params.fps_fixed is not None:
        fps = Params.fps_fixed
    else:
        fps = get_fps(Params.videoFileName)
    tot_frames = int(Params.tot_sec*fps)
    
    VideoCapture(Params.videoFileName, sharedData, fps=fps,
                       sleep=Params.fake_delay, resize=Params.resize)


    sig = []
    sig2 = []
    processed_frames_count = 0                              #第一張臉的frame數量
    processed_frames_count2 = 0                             #第二張臉的frame數量
    sig_buff_dim = int(fps * Params.winSize)                #30*6=180偵
    sig_stride = int(fps * Params.stride)                   #改window大小的地方1*30
    sig_buff_counter = sig_stride
    sig_buff_counter2 = sig_stride
    BPM_obj = None

    timeCount = []
    
    #testtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt
    x=0
    total = 0
    #testtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt
    
    send_images_count = 0
    send_images_stride = 3             #幾步更新一次畫面

    with mp_face_mesh.FaceMesh(
            max_num_faces=2,                #mediapipe 偵測多人臉
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while True:
            start_time = time.perf_counter()*1000
            frame = None
            if not sharedData.q_frames.empty():  # read frames from shared data
                frame = sharedData.q_frames.get()
                if type(frame) == int:  # cap stopped
                    break
            if not sharedData.q_stop.empty():  # GUI stopped
                sharedData.q_stop.get()
                break
            if frame is None:
                continue
            # convert the BGR image to RGB.
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # image = cv2.flip(image, 1)
            ##TEST##
            currentTime = None
            time1 = time.perf_counter()
            if not sharedData.q_times.empty():  # read frames from shared data
                currentTime = sharedData.q_times.get()
                Params.endTime = currentTime
                # print(currentTime)
                if fpsCount % fps == 0:
                    sharedData.q_saveTimes.put(currentTime)
                    # print(currentTime)

                    startTime = datetime.strptime(
                        Params.startTime, "%Y-%m-%d %H:%M:%S.%f")
                    currentTime = datetime.strptime(
                        currentTime, "%Y-%m-%d %H:%M:%S.%f")
                    passTime = currentTime - startTime

                fpsCount += 1

            cv2.putText(image, str(passTime) + " ms", (image.shape[1] - 350, image.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            ##TEST##

            
            width = image.shape[1]
            height = image.shape[0]
            # [landmarks, info], with info->x_center ,y_center, r, g, b
            ldmks = np.zeros((468, 5), dtype=np.float32)
            ldmks[:, 0] = -1.0
            ldmks[:, 1] = -1.0
            ldmks2 = np.zeros((468, 5), dtype=np.float32)
            ldmks2[:, 0] = -1.0
            ldmks2[:, 1] = -1.0
            magic_ldmks = []
            ### face landmarks ###

            results = face_mesh.process(image)
            
            if results.multi_face_landmarks:#/////////////////////////////////////////////////////改多人的區域
                if flag==1:
                    flag = 0
                else:
                    if len(results.multi_face_landmarks) >= 1:
                        processed_frames_count += 1
                        face_landmarks = results.multi_face_landmarks[0]
                        face_thread = threading.Thread(target = multi_face_landmarks_parallel,args = (image,face_landmarks,mp_drawing,width,height,ldmks,sharedData.q_first_face_ldmks,min_x,min_y,max_x,max_y,sharedData.q_face_range,skin_ex))
                        face_thread.daemon = False
                        face_thread.start()

                    if len(results.multi_face_landmarks) == 2:
                        processed_frames_count2 += 1
                        face_landmarks_2 = results.multi_face_landmarks[1]
                        face_thread2 = threading.Thread(target = multi_face_landmarks_parallel,args = (image,face_landmarks_2,mp_drawing,width,height,ldmks2,sharedData.q_second_face_ldmks,min_x2,min_y2,max_x2,max_y2,sharedData.q_face_range2,skin_ex))
                        face_thread2.daemon = False
                        face_thread2.start()
                    else:
                        cropped_skin_im2 = tf.zeros_like(image)
                        full_skin_im2 = tf.zeros_like(image)
                        sig2.clear()
                        processed_frames_count2 = 0
                        state2 = 0
                        currentBPM2 = 0
                    face_thread.join()
                    if len(results.multi_face_landmarks) == 2:
                        face_thread2.join()
                        
                    if not sharedData.q_first_face_ldmks.empty():
                        cropped_skin_im = sharedData.q_first_face_ldmks.get()
                        full_skin_im = sharedData.q_first_face_ldmks.get()
                        min_x = sharedData.q_face_range.get()
                        min_y = sharedData.q_face_range.get()
                        max_x = sharedData.q_face_range.get()
                        max_y = sharedData.q_face_range.get()
                    if not sharedData.q_second_face_ldmks.empty():
                        cropped_skin_im2 = sharedData.q_second_face_ldmks.get()
                        full_skin_im2 = sharedData.q_second_face_ldmks.get()
                        min_x2 = sharedData.q_face_range2.get()
                        min_y2 = sharedData.q_face_range2.get()
                        max_x2 = sharedData.q_face_range2.get()
                        max_y2 = sharedData.q_face_range2.get()

                        temp_image = cropped_skin_im2
                        cropped_skin_im2 = cropped_skin_im
                        cropped_skin_im = temp_image
                        temp_image = full_skin_im2
                        full_skin_im2 = full_skin_im
                        full_skin_im = temp_image
            else:
                cropped_skin_im = tf.zeros_like(image)
                full_skin_im = tf.zeros_like(image)
                cropped_skin_im2 = tf.zeros_like(image)
                full_skin_im2 = tf.zeros_like(image)
            
            if results.multi_face_landmarks:
                if len(results.multi_face_landmarks) == 2:
                    if processed_frames_count2 >= sig_buff_dim:
                        time2 = time.perf_counter()-time1
                        total += time2
                        x+=1
                        if x==300:
                            print(total/300)  
                            x=0 
                            total = 0
               
            skinlen = len(cropped_skin_im)
            if skinlen<72 or skinlen==len(image):               #抓到的臉部太少，重新計算
                processed_frames_count = 0
                processed_frames_count2 = 0
                sig.clear()
                sig2.clear()
                state = 0
                state2 = 0
                currentBPM = 0
                currentBPM2 = 0
                cv2.putText(image, 'face too small', (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:                   #append到fps*stride個frame，開始計算                          
                ### SIG ###
                if Params.approach == 'patches':#選patch跑這裡/////////////////////////////////////////////////////////
                    if Params.method['method_func'] ==  gpu_2SR:               #測試2SR,做padding把臉以外的值填充0
                        if Params.resize:                                       #TRUE則做resize不用padding
                            temp = cv2.resize(cropped_skin_im, (Params.resize_size, Params.resize_size), interpolation=cv2.INTER_AREA)
                        else:
                            temp = tf.pad(cropped_skin_im,[[0,full_skin_im.shape[0]-cropped_skin_im.shape[0]],[full_skin_im.shape[1]-cropped_skin_im.shape[1],0],[0,0]])
                        magic_ldmks =tf.constant(
                            ldmks[Params.landmarks_list], dtype=tf.dtypes.float32)
                    else:
                        magic_ldmks = np.array(
                            ldmks[Params.landmarks_list], dtype=np.float32)
                        temp = sig_ext_met(magic_ldmks, full_skin_im, ldmks_regions,
                                    np.int32(SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH))
                        temp = temp[:, 2:]  # keep only rgb mean
                elif Params.approach == 'holistic':
                    temp = sig_ext_met(cropped_skin_im, np.int32(
                        SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH))
                sig.append(temp)  

                if len(results.multi_face_landmarks) ==2:                           #第2張臉的signal提取
                    if Params.approach == 'patches':
                        if Params.method['method_func'] ==  gpu_2SR:               #測試2SR,做padding把臉以外的值填充0
                            if Params.resize:                                       #TRUE則做resize不用padding
                                temp = cv2.resize(cropped_skin_im2, (Params.resize_size, Params.resize_size), interpolation=cv2.INTER_AREA)
                            else:
                                temp = tf.pad(cropped_skin_im2,[[0,full_skin_im.shape[0]-cropped_skin_im2.shape[0]],[full_skin_im.shape[1]-cropped_skin_im.shape[1],0],[0,0]])
                            magic_ldmks =tf.constant(
                                ldmks[Params.landmarks_list], dtype=tf.dtypes.float32)
                        else:
                            magic_ldmks = np.array(
                                ldmks[Params.landmarks_list], dtype=np.float32)
                            temp = sig_ext_met(magic_ldmks, full_skin_im, ldmks_regions,
                                        np.int32(SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH))
                            temp = temp[:, 2:]  # keep only rgb mean
                    elif Params.approach == 'holistic':
                        temp = sig_ext_met(cropped_skin_im2, np.int32(
                            SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH))        
                    sig2.append(temp)     
                
                            
            if len(cropped_skin_im2)==480 or len(cropped_skin_im2)<72:
                processed_frames_count2 = 0
                sig2.clear()
                state2 = 0
                currentBPM2 = 0
            # visualize original, patches and skin
            if send_images_count == send_images_stride:
                send_images_count = 0
                ###TEST###先省略
                # idx: 59 = 234, idx: 7 = 10
                # idx: 80 = 345, idx: 45 = 152
                # height = int(ldmks[45, 0]) - int(ldmks[5,0])
                # leftup_label = (int(ldmks[25, 1]), int(ldmks[25,0]))
                # rightdown_label = (int(ldmks[80, 1]), int(ldmks[80, 0]))
                if results.multi_face_landmarks:
                    if len(results.multi_face_landmarks) ==2:
                        cv2.rectangle(image, (min_x2, min_y2), (max_x2, max_y2), (0, 255, 0), 2)
                        cv2.putText(image,'face 1',(min_x2-10,min_y2-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                        cv2.putText(image,'face 2',(min_x-10,min_y-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                        cv2.putText(image,'face 1',(min_x-10,min_y-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                min_x  = 1000
                min_y  = 1000
                max_x = 0
                max_y = 0
                min_x2  = 1000
                min_y2  = 1000
                max_x2 = 0
                max_y2 = 0

                m = 0
                loc5.append((int(ldmks[5, 1]), int(ldmks[5, 0])))
                mtx = np.array(loc5)
                # cv2.putText(image,st1+str(loc5[ctr-1]),(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,cv2.LINE_AA)
                if ctr >= 1:
                    m = ((int(mtx[ctr, 1])-int(mtx[ctr-1, 1]))**2 +
                         (int(mtx[ctr, 0])-int(mtx[ctr-1, 0]))**2)**(1/2)
                    cv2.putText(image, st2+str(round(m, 3)), (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    if m > movement:
                        cv2.putText(
                            image, st3, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
                ctr = ctr+1
                ###TEST###

                sharedData.q_video_image.put(image)
                if Params.visualize_skin == True:
                    if results.multi_face_landmarks:
                        if len(results.multi_face_landmarks)>1:
                            full_skin_im = np.add(np.array(full_skin_im),np.array(full_skin_im2))
                            sharedData.q_skin_image.put(full_skin_im)
                        else:
                            sharedData.q_skin_image.put(full_skin_im)
                    else:
                        sharedData.q_skin_image.put(full_skin_im)        #full_skin_im   cropped_skin_im
                    
                # if Params.approach == 'patches' and Params.visualize_landmarks == True:
                #     annotated_image = full_skin_im.copy()
                #     for idx in Params.landmarks_list:       #/////////////idx為mediapipe的468個點中的100個點
                #         cv2.circle(
                #             annotated_image, (int(ldmks[idx, 1]), int(ldmks[idx, 0])), radius=0, color=Params.font_color, thickness=-1)
                #         if Params.visualize_landmarks_number == True:
                #             cv2.putText(annotated_image, str(idx),
                #                         (int(ldmks[idx, 1]), int(ldmks[idx, 0])), cv2.FONT_HERSHEY_SIMPLEX, Params.font_size,  Params.font_color,  1)
                #     if Params.visualize_patches == True:
                #         if Params.patches == "squares":
                #             sides = [Params.squares_dim, ] * len(magic_ldmks)
                #             sides = np.array(sides)
                #             annotated_image = draw_rects(
                #                 annotated_image, np.array(magic_ldmks[:, 1]), np.array(magic_ldmks[:, 0]), sides, sides, color)
                #         elif Params.patches == "rects":
                #             rects_dims = np.array(Params.rects_dims)
                #             annotated_image = draw_rects(
                #                 annotated_image, np.array(magic_ldmks[:, 1]),
                #                 np.array(magic_ldmks[:, 0]), rects_dims[:,0],rects_dims[:,1] , color)
                #     # visualize patches
                #     sharedData.q_patches_image.put(annotated_image)
            else:
                send_images_count += 1


            if processed_frames_count >= sig_buff_dim:
                if len(sig) == sig_buff_dim:
                    sig = sig
                else:
                    sig = sig[1:]                   #取第2個到最後，也就是只取後30偵
                if sig_buff_counter == 2 :
                    flag = 1
                if sig_buff_counter == 1 :
                    sig_buff_counter = sig_stride           #算一次bvp後buffer重新堆30偵
                    state+=1
                    if Params.method['method_func'] != gpu_2SR:
                        copy_sig = np.array(sig, dtype=np.float32)
                        copy_sig = np.swapaxes(copy_sig, 0, 1)
                        copy_sig = np.swapaxes(copy_sig, 1, 2)

                        ### Pre_filtering ###////////////////////////////////////patch在這裡被去除的////////////////////
                        if Params.approach == 'patches':
                            copy_sig = rgb_filter_th(copy_sig, **{'RGB_LOW_TH':  np.int32(Params.color_low_threshold),
                                                                'RGB_HIGH_TH': np.int32(Params.color_high_threshold)})
                        for filt in Params.pre_filter:
                            if filt != {}:
                                if 'fps' in filt['params'] and filt['params']['fps'] == 'adaptive' and fps is not None:
                                    filt['params']['fps'] = float(fps)
                                if filt['params'] == {}:
                                    copy_sig = filt['filter_func'](
                                        copy_sig)
                                else:
                                    copy_sig = filt['filter_func'](
                                        copy_sig, **filt['params'])
                    
                    else:
                        copy_sig = np.array(sig, dtype=np.float32)
                    
                    ### BVP and BPM ###
                    
                    if state % interval==1:                     #interval 預設3秒
                        if Params.method['device_type'] == 'cpu': 
                            thread = threading.Thread(target = signals_to_bvps_to_bpm_cpu,args = (copy_sig,sharedData.q,Params.method['method_func'], BPM_obj,Params.method['params']))
                            thread.daemon = True
                            thread.start()                
                        elif Params.method['device_type'] == 'cuda':#執行這裡CHROM轉BVP//////////////////////////////////////////////
                            thread = threading.Thread(target = signals_to_bvps_to_bpm_cuda,args = (copy_sig,sharedData.q, Params.method['method_func'],BPM_obj, Params.method['params']))
                            thread.daemon = True
                            thread.start()
                        bpm = currentBPM
                    elif state % interval==0:
                        thread.join()
                        bpm =sharedData.q.get()
                        if state<=12:
                            foursecbpm.append(bpm)
                    else :
                        bpm = currentBPM
                        
                    if state>12:
                        if bpm>currentBPM+10:
                            if bpm>currentBPM+20:
                                bpm = currentBPM
                            else:
                                bpm = (currentBPM+bpm)/2
                        elif bpm<currentBPM-10:
                            if bpm<currentBPM-20:
                                bpm = currentBPM
                            else:
                                bpm = (currentBPM+bpm)/2
                    elif state==12 :
                        currentBPM = np.mean(foursecbpm)
                        if bpm>currentBPM+10:
                            if bpm>currentBPM+25:
                                bpm = currentBPM
                            else:
                                bpm = (currentBPM+bpm)/2
                        elif bpm<currentBPM-10:
                            if bpm<currentBPM-25:
                                bpm = currentBPM
                            else:
                                bpm = (currentBPM+bpm)/2
                        
                        
                    
                    # bpm_data = pd.Series(bpm)
                    # if os.path.exists('C:\\Users\\user\\Desktop\\bpm_data.csv'):    
                    #     old_bpm_data = pd.read_csv('C:\\Users\\user\\Desktop\\bpm_data.csv')
                    #     bpm_data = pd.concat([bpm_data],axis =0,ignore_index = True)    #bpm_data_sec,bpm_median
                    #     pd.concat([old_bpm_data,bpm_data],axis =1).to_csv('C:\\Users\\user\\Desktop\\bpm_data.csv',index = False)
                    # else:
                    #     bpm_data = pd.concat([bpm_data],axis =0,ignore_index = True).to_csv('C:\\Users\\user\\Desktop\\bpm_data.csv',index = False)
                    

                        
                    sharedData.q_bpm.put(bpm)
                    ###TEST###
                    currentBPM = np.round(bpm, 2)

                    temp = sharedData.q_saveTimes.get()
                    bpmTime.append(temp)
                    ###TEST###
                else:
                    sig_buff_counter -= 1
                
            if processed_frames_count2 >= sig_buff_dim:             #第2張臉
                if len(sig2) == sig_buff_dim:
                    sig2 = sig2
                else:
                    sig2 = sig2[1:]
                if sig_buff_counter2 == 1 :
                    sig_buff_counter2 = sig_stride           #算一次bvp後buffer重新堆30偵
                    state2+=1
                    if Params.method['method_func'] != gpu_2SR:
                        # sig_buff_counter = sig_stride
                        copy_sig = np.array(sig2, dtype=np.float32)
                        copy_sig = np.swapaxes(copy_sig, 0, 1)
                        copy_sig = np.swapaxes(copy_sig, 1, 2)

                        ### Pre_filtering ###////////////////////////////////////patch在這裡被去除的////////////////////
                        if Params.approach == 'patches':
                            copy_sig = rgb_filter_th(copy_sig, **{'RGB_LOW_TH':  np.int32(Params.color_low_threshold),
                                                                'RGB_HIGH_TH': np.int32(Params.color_high_threshold)})
                        for filt in Params.pre_filter:
                            if filt != {}:
                                if 'fps' in filt['params'] and filt['params']['fps'] == 'adaptive' and fps is not None:
                                    filt['params']['fps'] = float(fps)
                                if filt['params'] == {}:
                                    copy_sig = filt['filter_func'](
                                        copy_sig)
                                else:
                                    copy_sig = filt['filter_func'](
                                        copy_sig, **filt['params'])
                    
                    else:
                        copy_sig = np.array(sig2, dtype=np.float32)
                    
                    ## BVP ###
                    if state2 % interval==1:                     #interval 預設3秒
                        if Params.method['device_type'] == 'cpu': 
                            thread2 = threading.Thread(target = signals_to_bvps_to_bpm_cpu,args = (copy_sig,sharedData.q2, Params.method['method_func'],BPM_obj, Params.method['params']))
                            thread2.daemon = True
                            thread2.start()                
                        elif Params.method['device_type'] == 'cuda':#執行這裡CHROM轉BVP//////////////////////////////////////////////
                            thread2 = threading.Thread(target = signals_to_bvps_to_bpm_cuda,args = (copy_sig,sharedData.q2,Params.method['method_func'],BPM_obj, Params.method['params']))
                            thread2.daemon = True
                            thread2.start()
                        bpm = currentBPM2
                    elif state2 % interval==0:
                        thread2.join()
                        bpm =sharedData.q2.get() 
                    else :
                        bpm = currentBPM2
                    if state2>10:
                        if bpm>currentBPM+10:
                            if bpm>currentBPM+15:
                                bpm = currentBPM
                            else:
                                bpm = (currentBPM+bpm)/2
                        elif bpm<currentBPM-10:
                            if bpm<currentBPM-15:
                                bpm = currentBPM
                            else:
                                bpm = (currentBPM+bpm)/2
                    sharedData.q_bpm2.put(bpm)
                    currentBPM2 = np.round(bpm, 2)
                    
                    
                else:
                    sig_buff_counter2 -= 1       
 
            ###TEST###
            cv2.putText(image, "BPM : " + str(currentBPM), (20, image.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 53, 53), 2, cv2.LINE_AA)
            
            if sig2:
                cv2.putText(image, "BPM2: " + str(currentBPM2), (20, image.shape[0] - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 53, 53), 2, cv2.LINE_AA)

            cv2.putText(image, "skinlen: " + str(skinlen), (image.shape[1] - 220, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ###TEST###

            end_time = time.perf_counter()*1000
            timeCount.append(end_time-start_time)
            if len(timeCount) > 100:
                timeCount = timeCount[1:]
            
            ### loop break ###
            if tot_frames is not None and tot_frames > 0 and processed_frames_count >= tot_frames:
                break
    print('out check point')  
    if len(timeCount) > 2:
        print("Times in milliseconds of the computation of a frame:")
        print("mean:   ",statistics.mean(timeCount))
        print("median: ",statistics.median(timeCount))
        print("max:    ",max(timeCount))
        print("min:    ",min(timeCount))
    return


"""
if __name__ == "__main__":
    sd = SharedData()
    Params.videoFileName = "/home/frea/Documents/VHR/LGI_PPGI/lgi_alex/alex_resting/cv_camera_sensor_stream_handler.avi"
    Params.tot_sec = 0
    t = Thread(target=VHRroutine, args=(sd,))
    t.start()
    t.join()
"""

import os
import sys
sys.path.append("D://rPPG//FCUAI_rPPG_workplace//pyVHR")
import numpy as np
import pandas as pd
import cupy
import scipy
from scipy.signal import butter
from params import Params
from BPM.BPM import BPMcuda,BPM
import tensorflow as tf
from BVP.model import *

def NNsignals_to_bvps_to_bpm_cpu(sig,queue):
    model = EfficientPhys_attention(frame_depth = 180,img_size=72,in_channels=3) #EfficientPhys_residualTSM   EfficientPhys_attention
    model.eval()
    model.load_state_dict(torch.load('realtime//EfficientPhys_model155.pt',map_location=torch.device('cpu')))
    sig = sig.reshape(-1,3,72,72)
    outputs = model(torch.tensor(sig))
    outputs = outputs[:,0]
    outputs = outputs.detach().numpy()
    # bvp_data = pd.Series(outputs)
    # if os.path.exists('C:\\Users\\user\\Desktop\\bvp_data.csv'):    
    #     old_bvp_data = pd.read_csv('C:\\Users\\user\\Desktop\\bvp_data.csv')
    #     bvp_data = pd.concat([bvp_data],axis =0,ignore_index = True)    #bpm_data_sec,bpm_median
    #     pd.concat([old_bvp_data,bvp_data],axis =1).to_csv('C:\\Users\\user\\Desktop\\bvp_data.csv',index = False)
    # else:
    #     bvp_data = pd.concat([bvp_data],axis =0,ignore_index = True).to_csv('C:\\Users\\user\\Desktop\\bvp_data.csv',index = False)
    [b_pulse, a_pulse] = butter(1, [0.75 / 30 * 2, 3 / 30 * 2], btype='bandpass')
    outputs = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(outputs))
    fft = np.abs(scipy.fft.rfft(outputs,n = 1024))
    bpm_place = np.argmax(fft)
    bpm = scipy.fft.rfftfreq(1024,1/30)
    bpm = bpm[bpm_place]*60
    queue.put(bpm)

def signals_to_bvps_to_bpm_cuda(sig,queue, gpu_method,BPM_obj, params={}):
    """
    Transform an input RGB signal in a BVP signal using a rPPG method (see pyVHR.BVP.methods).
    This method must use cupy and executes on GPU. You can pass also non-RGB signal but the method used must handle its shape.

    Args:
        sig(float32 tensor)
        sig (float32 ndarray): RGB Signal as float32 ndarray with shape  [num_estimators, rgb_channels, num_frames].
            You can pass also a generic signal but the method used must handle its shape and type.
        cupy_method: a method that comply with the fucntion signature documented in pyVHR.BVP.methods. This method must use Cupy.
        params (dict): dictionary of usefull parameters that will be passed to the method.
    
    Returns:
        float32 ndarray: BVP signal as float32 ndarray with shape [num_estimators, num_frames].
    """
    if sig.shape[0] == 0:
        bvp = np.zeros((0, sig.shape[2]), dtype=sig.dtype)
    else:
        gpu_sig = cupy.asarray(sig)       
        if len(params) > 0:#/////////////////////////////////////////////cupy_method = CHROM或其他方法
            bvps = gpu_method(gpu_sig, **params)
        else:
            bvps = gpu_method(gpu_sig)         #CHROM走這裡
        bvps = cupy.asnumpy(bvps)
        bvp = bvps
    ### Post_filtering ###
    fps = Params.fps_fixed
    for filt in Params.pre_filter:
        if filt != {}:
            bvp = tf.expand_dims(bvp, axis=1)
            if 'fps' in filt['params'] and filt['params']['fps'] == 'adaptive' and fps is not None:
                filt['params']['fps'] = float(fps)
            if filt['params'] == {}:
                bvp = filt['filter_func'](bvp)
            else:
                bvp = filt['filter_func'](
                    bvp, **filt['params'])
            bvp = tf.squeeze(bvp, axis=1)
    
    ### BPM ###
    if Params.cuda:
        bvp_device = np.asarray(bvp)
        if BPM_obj == None:
            last_bpm = 0
            bpm_num = 0
            BPM_obj = BPMcuda(bvp_device, fps,
                            minHz=Params.minHz, maxHz=Params.maxHz)
        else:
            BPM_obj.data = bvp_device
        if Params.BPM_extraction_type == "welch":
            bpm = BPM_obj.BVP_to_BPM()
            bpm = cupy.asnumpy(bpm)
        elif Params.BPM_extraction_type == "psd_clustering":
            bpm = BPM_obj.BVP_to_BPM_PSD_clustering()
    else:
        if BPM_obj == None:
            BPM_obj = BPM(bvp, fps, minHz=Params.minHz,
                        maxHz=Params.maxHz)
        else:
            BPM_obj.data = bvp
        if Params.BPM_extraction_type == "welch":
            bpm = BPM_obj.BVP_to_BPM()
        elif Params.BPM_extraction_type == "psd_clustering":
            bpm = BPM_obj.BVP_to_BPM_PSD_clustering()
    if Params.approach == 'patches':  # Median of multi BPMs
        if len(bpm.shape) > 0 and bpm.shape[0] == 0:
            bpm = np.float32(0.0)
        else:
            bpm = np.float32(np.median(bpm))
    queue.put(bpm)

def signals_to_bvps_to_bpm_cpu(sig,queue, cpu_method, BPM_obj,params={}):
    """
    Transform an input RGB signal in a BVP signal using a rPPG 
    method (see pyVHR.BVP.methods).
    This method must use and execute on CPU.
    You can pass also non-RGB signal but the method used must handle its shape.

    Args:
        sig (float32 ndarray): RGB Signal as float32 ndarray with shape  [num_estimators, rgb_channels, num_frames].
            You can pass also a generic signal but the method used must handle its shape and type.
        cpu_method: a method that comply with the fucntion signature documented 
            in pyVHR.BVP.methods. This method must use Numpy.
        params (dict): dictionary of usefull parameters that will be passed to the method.
    
    Returns:
        float32 ndarray: BVP signal as float32 ndarray with shape [num_estimators, num_frames].
    """
    if sig.shape[0] == 0:
        bvp = np.zeros((0, sig.shape[2]), dtype=sig.dtype)
    else:
        cpu_sig = np.array(sig)
        if len(params) > 0:             #//////////////cupy_method = SSR走cpu
            bvps = cpu_method(cpu_sig, **params)
        else:
            bvps = cpu_method(cpu_sig)
        bvp = bvps
    ### Post_filtering ###
    fps = Params.fps_fixed
    for filt in Params.pre_filter:
        if filt != {}:
            bvp = tf.expand_dims(bvp, axis=1)
            if 'fps' in filt['params'] and filt['params']['fps'] == 'adaptive' and fps is not None:
                filt['params']['fps'] = float(fps)
            if filt['params'] == {}:
                bvp = filt['filter_func'](bvp)
            else:
                bvp = filt['filter_func'](
                    bvp, **filt['params'])
            bvp = tf.squeeze(bvp, axis=1)
    
    ### BPM ###
    if Params.cuda:
        bvp_device = np.asarray(bvp)
        if BPM_obj == None:
            last_bpm = 0
            bpm_num = 0
            BPM_obj = BPMcuda(bvp_device, fps,
                            minHz=Params.minHz, maxHz=Params.maxHz)
        else:
            BPM_obj.data = bvp_device
        if Params.BPM_extraction_type == "welch":
            bpm = BPM_obj.BVP_to_BPM()
            bpm = cupy.asnumpy(bpm)
        elif Params.BPM_extraction_type == "psd_clustering":
            bpm = BPM_obj.BVP_to_BPM_PSD_clustering()
    else:
        if BPM_obj == None:
            BPM_obj = BPM(bvp, fps, minHz=Params.minHz,
                        maxHz=Params.maxHz)
        else:
            BPM_obj.data = bvp
        if Params.BPM_extraction_type == "welch":
            bpm = BPM_obj.BVP_to_BPM()
        elif Params.BPM_extraction_type == "psd_clustering":
            bpm = BPM_obj.BVP_to_BPM_PSD_clustering()
    if Params.approach == 'patches':  # Median of multi BPMs
        if len(bpm.shape) > 0 and bpm.shape[0] == 0:
            bpm = np.float32(0.0)
        else:
            bpm = np.float32(np.median(bpm))
    queue.put(bpm)

def multi_face_landmarks_parallel(image,face_landmarks,mp_drawing,width,height,ldmks,queue,min_x,min_y,max_x,max_y,q_face_range,skin_ex):
    landmarks = [l for l in face_landmarks.landmark]
    for idx in range(len(landmarks)):       #/////////////idx為mediapipe的468個點
        landmark = landmarks[idx]
        if not ((landmark.HasField('visibility') and landmark.visibility < 0.5)
                or (landmark.HasField('presence') and landmark.presence < 0.5)):
            coords = mp_drawing._normalized_to_pixel_coordinates(
                landmark.x, landmark.y, width, height)
            if coords:
                ldmks[idx, 0] = coords[1]
                ldmks[idx, 1] = coords[0]
                ###TEST###
                if(min_x > coords[0]):
                    min_x = coords[0]
                if(min_y > coords[1]):
                    min_y = coords[1]
                if(max_x < coords[0]):
                    max_x = coords[0]
                if(max_y < coords[1]):
                    max_y = coords[1]
    
    cropped_skin_im, full_skin_im = skin_ex.extract_skin(
                        image, ldmks)
    queue.put(cropped_skin_im)
    queue.put(full_skin_im)
    q_face_range.put(min_x)
    q_face_range.put(min_y)
    q_face_range.put(max_x)
    q_face_range.put(max_y)
    
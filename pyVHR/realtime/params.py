import sys
sys.path.append("D://rPPG//FCUAI_rPPG_workplace//pyVHR")
from pyVHR.extraction.utils import MagicLandmarks
from pyVHR.BVP.methods import *
from pyVHR.BVP.filters import *
from pyVHR.extraction.utils import MagicLandmarks

class Params:
    # DEFAULT PARAMETERS
    startTime = None
    endTime = None
    methodName = ""


    videoFileName = ""
    fps_fixed = 30
    tot_sec = 0
    winSize = 6             #winsize基本上代表秒數
    stride = 1
    cuda = True
    skin_extractor = 'convexhull'  # or faceparsing
    approach = 'patches'  # or holistic
    patches = 'squares'  # or rects
    type = 'mean'
    landmarks_list = MagicLandmarks.equispaced_facial_points
    squares_dim = 30.0
    rects_dims = []
    # extractor params
    skin_color_low_threshold = 75
    skin_color_high_threshold = 230
    sig_color_low_threshold = 75
    sig_color_high_threshold = 230
    # rgb_filter_th params
    color_low_threshold = 75
    color_high_threshold = 230
    # visualize skin and patches
    visualize_skin = True
    visualize_patches = True
    visualize_landmarks = True
    visualize_landmarks_number = True
    font_size = 0.3
    font_color = (255, 0, 0, 255)

    # Pre filtering
    # dictionary of {filter_func, params}
    pre_filter = [{'filter_func': BPfilter, 'params': {
        'minHz': 0.65, 'maxHz': 4.0, 'fps': 'adaptive', 'order': 6}}]

    # BVP method
    # dictionary of {method_func, device_type, params}
    method = {'method_func': cpu_SSR,                    #cpu_SSR、cupy_CHROM
              'device_type': 'cpu', 'params': {}}          #'device_type': 'cuda'

    # Post filtering
    # dictionary of {filter_func, params}
    post_filter = [{'filter_func': BPfilter, 'params': {
        'minHz': 0.65, 'maxHz': 4.0, 'fps': 'adaptive', 'order': 6}}]

    # BPM params
    minHz = 0.75
    maxHz = 4.0
    # WELCH: CPU, GPU
    # PSD_CLUSTERING: CPU, GPU
    # USE psd_clustering only with patches!
    BPM_extraction_type = "welch" # or 'psd_clustering'


    # Utils
    fake_delay = False
    resize = True
    out_path = None

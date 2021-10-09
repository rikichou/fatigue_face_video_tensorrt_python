from . import fatigue

test_video_path = '/home/ruiming/workspace/pro/fatigue/data/test/video/lookdown_face/face_02_03_0_0_28716582.avi'

#model_path = '/home/ruiming/workspace/pro/source/mmaction2/work_dirs/fatigue_r50_clean/fatigue_r50_clean_fp16.trt'
model_path = '/home/ruiming/workspace/pro/source/mmaction2/work_dirs/fatigue_r50_clean_withnormal/fatigue_r50_clean_withnormal_fp16.trt'
config_path = '/home/ruiming/workspace/pro/source/mmaction2/configs/recognition/csn/fatigue_r50_clean_inference.py'

fh = fatigue.FatigueFaceVideoTensorrt(model_path, config_path)
results = fh(test_video_path)

print(results)
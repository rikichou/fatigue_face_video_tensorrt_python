import argparse
import fatigue

test_video_path = '/home/ruiming/workspace/pro/fatigue/data/test/video/lookdown_face/face_02_03_0_0_28716582.avi'

#model_path = '/home/ruiming/workspace/pro/source/mmaction2/work_dirs/fatigue_r50_clean/fatigue_r50_clean_fp16.trt'
model_path = '/home/ruiming/workspace/pro/source/mmaction2/work_dirs/fatigue_r50_clean_withnormal/fatigue_r50_clean_withnormal_fp16.trt'
config_path = '/home/ruiming/workspace/pro/source/mmaction2/configs/recognition/csn/fatigue_r50_clean_inference.py'

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('test_video_path',
                        type=str,
                        help='face video path')
    parser.add_argument('--model_path', default='model/fatigue_r50_clean_withnormal/fatigue_r50_clean_withnormal_fp16.trt', type=str, help='tensorrt model path')
    parser.add_argument('--config_path', default='model/fatigue_r50_clean_withnormal/fatigue_r50_clean_inference.py', type=str, help='mmaction inference config path')
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2, 3],
        default=3,
        help='directory level of data')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=8,
        help='number of workers to build rawframes')
    parser.add_argument(
        '--ext',
        type=str,
        default='avi',
        choices=['avi', 'mp4', 'webm'],
        help='video file extensions')
    parser.add_argument(
        '--mixed-ext',
        action='store_true',
        help='process video files with mixed extensions')
    parser.add_argument(
        '--new-width', type=int, default=0, help='resize image width')
    parser.add_argument(
        '--new-height', type=int, default=0, help='resize image height')
    parser.add_argument(
        '--new-short',
        type=int,
        default=0,
        help='resize image short side length keeping ratio')

    args = parser.parse_args()

    return args

args = parse_args()

fh = fatigue.FatigueFaceVideoTensorrt(args.model_path, args.config_path)
results = fh(args.test_video_path)

print(results)
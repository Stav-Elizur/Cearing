import cv2
from pose_format import PoseHeader, Pose
from pose_format.utils.holistic import load_holistic
from pose_format.utils.reader import BufferReader

from utils.pose_utils import pose_hide_legs

if __name__ == '__main__':

    def load_video_frames(cap: cv2.VideoCapture):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()


    # Load video frames
    print('Loading video ...')
    cap = cv2.VideoCapture('original1.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = load_video_frames(cap)

    # Perform pose estimation
    print('Estimating pose ...')
    pose = load_holistic(frames,
                         fps=fps,
                         width=width,
                         height=height,
                         progress=True,
                         additional_holistic_config={'model_complexity': 2})


    # Write
    print('Saving to disk ...')
    with open('output5.pose', "wb") as f:
        pose.write(f)
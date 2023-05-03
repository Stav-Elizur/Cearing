import cv2
import numpy as np
from pose_format import PoseHeader, Pose
from pose_format.pose_header import PoseNormalizationInfo
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.reader import BufferReader
from tqdm import tqdm

from utils.constants import DEFAULT_COMPONENTS
from moviepy.editor import *
from utils.pose_utils import pose_hide_legs

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def pose_normalization_info(pose_header: PoseHeader) -> PoseNormalizationInfo:
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                                              p2=("POSE_LANDMARKS", "LEFT_SHOULDER"))

    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(p1=("BODY_135", "RShoulder"), p2=("BODY_135", "LShoulder"))

    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(p1=("pose_keypoints_2d", "RShoulder"),
                                              p2=("pose_keypoints_2d", "LShoulder"))

    raise ValueError("Unknown pose header schema for normalization")


def visualize_pose(pose: Pose, pose_name: str):
    # Draw original pose
    visualizer = PoseVisualizer(pose, thickness=2)

    visualizer.save_video(pose_name, visualizer.draw(),
                          custom_ffmpeg="C:\projects\\ffmpeg\\bin")


def save_video(name: str):
    print('Loading input pose ...')
    with open(f'{name}.pose', 'rb') as pose_file:
        pose = Pose.read(pose_file.read())
        pose = pose.get_components(DEFAULT_COMPONENTS)
        pose_hide_legs(pose)

        print('Generating videos ...')
        visualize_pose(pose, f'{name}.mp4')


def concate_two_videos(first_video_name: str,
                       second_video_name: str,
                       final_video_name: str):
    # Load the two video files
    first_video = VideoFileClip(f'{first_video_name}.mp4')
    second_video = VideoFileClip(f'{second_video_name}.mp4')

    # Concatenate the clips
    final_video = concatenate_videoclips([first_video, second_video])

    # Write the final clip to a new file
    final_video.write_videofile(f'{final_video_name}.mp4')


def save_frame():
    import cv2

    # Read in the video
    cap = cv2.VideoCapture('video.mp4')

    # Set the position of the frame you want to save (in this case, the first frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Read the frame
    ret, frame = cap.read()

    if ret:
        # Save the frame as an image
        cv2.imwrite('frame.jpg', frame)

    # Release the video capture object
    cap.release()

def show_highest_red_pixel_position(photo):
    import numpy as np

    # Extract the red channel
    red_channel = photo[:, :, 2]

    # Find the maximum value and its location in the red channel
    max_val = np.max(red_channel)
    max_loc = np.argmax(red_channel)

    # Convert the index to x,y coordinates
    height, width, channels = photo.shape
    x = max_loc % width
    y = max_loc // width

    # Print the coordinates of the highest red pixel
    print("Coordinates of highest red pixel: ({}, {})".format(x, y))

    return x, y


def fix_video(name: str):
    # Read the photo
    photo = cv2.imread('init_frame.jpg')

    init_pose_x, init_pose_y = show_highest_red_pixel_position(photo)

    # Read in the original video
    cap = cv2.VideoCapture(f'{name}.mp4')

    # Get the total number of frames in the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the last frame of the video
    _, first_frame = cap.read()

    video_init_pose_x, video_init_pose_y = show_highest_red_pixel_position(first_frame)

    diff_x, diff_y = init_pose_x - video_init_pose_x, init_pose_y - video_init_pose_y

    # Reset the video's position
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Create a VideoWriter object to write the modified frames to a new video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'fixed_{name}.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Loop over each frame of the second video
    for i in tqdm(range(15)):
        ret, frame = cap.read()

        if ret:
            # get height and width of frame
            height, width, channels = frame.shape

            # loop through x and y coordinates
            for y in range(height):
                for x in range(width):
                    if 0 <= (y + diff_y) < height:
                        frame[y + diff_y, x] = frame[y, x]
                    if 0 <= (x + diff_x) < width:
                        frame[y, x + diff_x] = frame[y, x]

            # Write the warped frame to the output video file
            out.write(frame)
        else:
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()


def conc():
    import cv2
    import numpy as np

    # Load the first video
    cap1 = cv2.VideoCapture('output.mp4')

    # Get the frame rate and frame size of the first video
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load the second video
    cap2 = cv2.VideoCapture('output1.mp4')

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('final_output.mp4', fourcc, fps, (width, height))


    # Read the last frame of the first video
    _, frame1 = cap1.read()
    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        out.write(frame)
        # frame1 = frame

    # Loop over each frame of the second video and write it to the output video
    for i in tqdm(range(10)):
        ret, frame = cap2.read()
        if not ret:
            break
        height, width, _ = frame.shape

        # Loop over each pixel in the image
        for x in range(width):
            for y in range(height):
                # Calculate the new y-coordinate for the pixel
                new_y = int(y - 100)

                # Check if the new y-coordinate is within the bounds of the image
                if new_y < height:
                    # Set the pixel at the new location to the value of the original pixel
                    frame[new_y, x, :] = frame[y, x, :]
        out.write(frame)

    # Release the input and output videos
    cap1.release()
    cap2.release()
    out.release()


if __name__ == '__main__':
    conc()
    # save_video('output')
    # save_video('output1')
    # concate_two_videos('output', 'output1', 'final')
    # from IPython.display import Image
    #
    # with open(f'output1.pose', 'rb') as pose_file:
    #     pose = Pose.read(pose_file.read())
    #     pose = pose.get_components(DEFAULT_COMPONENTS)
    #     pose_hide_legs(pose)
    #     pose = pose.augment2d(rotation_std=0, shear_std=1, scale_std=0.7)
    #
    #     visualizer = PoseVisualizer(pose, thickness=2)
    #
    #     visualize_pose(pose, f'test.mp4')
    #
    #     # display(Image(open('test.gif', 'rb').read()))

    # import cv2
    # import numpy as np
    #
    # # Load the image
    # img = cv2.imread('init_frame1.jpg')
    #
    # # Get the dimensions of the image
    # height, width, _ = img.shape
    #
    # # Define the ratio to move the pixels in the y-axis
    # y_ratio = -0.5
    #
    # # Create a new array to store the shifted image
    # shifted_img = np.zeros_like(img)
    #
    # # Loop over each pixel in the image
    # for x in range(width):
    #     for y in range(height):
    #         # Calculate the new y-coordinate for the pixel
    #         new_y = int(y - 150)
    #
    #         # Check if the new y-coordinate is within the bounds of the image
    #         if new_y < height:
    #             # Set the pixel at the new location to the value of the original pixel
    #             shifted_img[new_y, x, :] = img[y, x, :]
    #
    # # Display the original and shifted images side by side
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Shifted Image', shifted_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
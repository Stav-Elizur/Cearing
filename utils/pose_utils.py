from pose_format import PoseHeader, Pose
from pose_format.pose_header import PoseNormalizationInfo


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


def pose_hide_legs(pose: Pose):
    if pose.header.components[0].name == "POSE_LANDMARKS":
        point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        points = [
            pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
            for n in point_names
            for side in ["LEFT", "RIGHT"]
        ]
        pose.body.confidence[:, :, points] = 0 # Confidence Shape (Frames, People, Points)
        pose.body.data[:, :, points, :] = 0 # Data Shape (Frames, People, Points, Dims)
    else:
        raise ValueError("Unknown pose header schema for hiding legs")

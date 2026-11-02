from dataclasses import dataclass
import math

from processing.pose_estimation import Landmark, PoseFrame


@dataclass(frozen=True)
class JointAngle:
    """1つの関節角度。"""

    name: str
    degrees: float
    frame_index: int


JOINT_ANGLE_DEFINITIONS: dict[str, tuple[str, str, str]] = {
    "左肘": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "右肘": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "左肩": ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
    "右肩": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
    "左膝": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "右膝": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    "左股関節": ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
    "右股関節": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
}


def calculate_angle(
    point_a: Landmark,
    vertex: Landmark,
    point_c: Landmark,
) -> float:
    """3点の座標から、vertexを頂点とする角度を度数で計算する。"""
    vector_a = (
        point_a.x - vertex.x,
        point_a.y - vertex.y,
        point_a.z - vertex.z,
    )
    vector_c = (
        point_c.x - vertex.x,
        point_c.y - vertex.y,
        point_c.z - vertex.z,
    )

    length_a = math.sqrt(sum(value * value for value in vector_a))
    length_c = math.sqrt(sum(value * value for value in vector_c))
    if length_a == 0 or length_c == 0:
        raise ValueError("角度計算に必要な点が重なっています。")

    cosine = sum(
        left * right for left, right in zip(vector_a, vector_c)
    ) / (length_a * length_c)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def calculate_frame_angles(
    pose_frame: PoseFrame,
    min_visibility: float = 0.5,
) -> tuple[JointAngle, ...]:
    """1フレームの骨格から計算可能な関節角度を返す。"""
    if not 0 <= min_visibility <= 1:
        raise ValueError("min_visibilityは0から1の範囲で指定してください。")

    if pose_frame.landmarks is None:
        return ()

    landmarks = {landmark.name: landmark for landmark in pose_frame.landmarks}
    angles: list[JointAngle] = []

    for name, landmark_names in JOINT_ANGLE_DEFINITIONS.items():
        points = [landmarks.get(landmark_name) for landmark_name in landmark_names]
        if any(point is None for point in points):
            continue
        if any(point.visibility < min_visibility for point in points if point is not None):
            continue

        try:
            degrees = calculate_angle(points[0], points[1], points[2])
        except ValueError:
            continue
        angles.append(
            JointAngle(
                name=name,
                degrees=degrees,
                frame_index=pose_frame.frame_index,
            )
        )

    return tuple(angles)


def calculate_all_frame_angles(
    pose_frames: tuple[PoseFrame, ...],
    min_visibility: float = 0.5,
) -> tuple[tuple[JointAngle, ...], ...]:
    """動画の全検出フレームについて関節角度を計算する。"""
    return tuple(
        calculate_frame_angles(pose_frame, min_visibility=min_visibility)
        for pose_frame in pose_frames
    )

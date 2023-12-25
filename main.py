import copy
import math
import numpy as np
import cv2


def detect_april_tags(image: np.ndarray) -> tuple[list[list[list[int]]], list[int]]:
    """
    :param image: the image where we want to detect
    :return: an array of the detected april tags and an array of their id's
    """
    processed_frame = copy.deepcopy(image)
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.DICT_APRILTAG_16h5
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco_dict), parameters)
    proj_squares, ids, rejected_img_points = detector.detectMarkers(processed_frame)
    if (proj_squares is ()):
        return [], []
    return [a[0] for a in proj_squares], [a[0] for a in ids]


def find_projected_tag_center(tag: list[list[int or float]]) -> list[int or float]:
    """
    :param tag: a 2d list where each point is a point on the tag as detected by the detector
    :return: a list that represents the 2d vector of the tag's center coordinates on the image
    """
    p1 = tag[0]
    p2 = tag[1]
    p3 = tag[2]
    p4 = tag[3]
    line1 = (p1, p3)
    line2 = (p2, p4)
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]


def screen_point_to_normalized_vector(point: list[int or float], width: int, height: int,
                                      focal_length_x: float, focal_length_y: float) -> np.ndarray:
    v = np.array([(point[0] - (width * 0.5)) / focal_length_x,
                  (point[1] - (height * 0.5)) / focal_length_y,
                  1])
    v /= np.linalg.norm(v)
    return v


def diagonal_to_camera_oriented(p1_normalized: np.ndarray, p2_normalized: np.ndarray,
                                center_normalized: np.ndarray, diagonal_length: float) -> tuple[np.ndarray, np.ndarray]:
    """
    :param p1_normalized:  the first point normalized to 1
    :param p2_normalized:  the oposite point to p1 normalized to 1
    :param center_normalized:  the center point normalized to 1
    :param diagonal_length: the length of the diagonal
    :return: correctly scaled p1 and p2 to fit real world camera oriented coordinates
    """
    alpha_1 = math.acos(np.dot(p1_normalized, center_normalized))  # the angle between the center and p1
    alpha_2 = math.acos(np.dot(p2_normalized, center_normalized))  # the angle between center and p2

    p2_to_p1_ratio = math.sin(alpha_1) / math.sin(alpha_2)
    p2 = p2_normalized * p2_to_p1_ratio
    p1 = p1_normalized
    scale = diagonal_length / np.linalg.norm(p2 - p1)
    p1 *= scale
    p2 *= scale
    return np.append(p1, 1), np.append(p2, 1)


def tag_projected_points_to_camera_oriented(tag: list[list[int or float]],
                                            width: int, height: int, diagonal_length: float,
                                            focal_length_x: float, focal_length_y: float) -> np.ndarray:
    """
    :param tag: the tag corners as projected in the image in the same order as they were identified
    :param width: the images's width in pixels
    :param height: the image's height in pixels
    :param diagonal_length: the length of the tags diagonal in real life
    :param focal_length_x: the focal length in the x axis (tan(fov_x/2)*2) * width
    :param focal_length_y: the focal length in the y axis (tan(fov_y/2)*2) * height
    :return: a 4*4 matrix of each corners position as in the same order as the original tag with the last row being 1's
    """
    normalized_corners = [screen_point_to_normalized_vector(corner, width, height, focal_length_x, focal_length_y)
                          for corner in tag]
    normalized_center = screen_point_to_normalized_vector(find_projected_tag_center(tag),
                                                          width, height, focal_length_x, focal_length_y)
    p1, p3 = diagonal_to_camera_oriented(normalized_corners[0], normalized_corners[2],
                                         normalized_center, diagonal_length)
    p2, p4 = diagonal_to_camera_oriented(normalized_corners[1], normalized_corners[3],
                                         normalized_center, diagonal_length)
    return np.column_stack([p1, p2, p3, p4])


def camera_oriented_to_extrinsic_matrix(camera_oriented_tag_matrix: np.ndarray, field_oriented_tag_inverse_matrix) \
        -> np.ndarray:
    """
    :param camera_oriented_tag_matrix:  the 4*4 matrix of the camera oriented position of the each corner tag with the
    order being the same as the order the points were detected and the last row being all 1's
    :param field_oriented_tag_inverse_matrix: the inverse matrix of the 4*4 matrix that represents the field oriented
    position of each corner of the tag with the order being the same as the order they are detected and the last row
    being all 1's, you can compute this matrix one time for every tag before runtime as it is constant for each tag
    :return: the camera extrinsic matrix (4*4)
    """
    return camera_oriented_tag_matrix @ field_oriented_tag_inverse_matrix

def extrinsic_matrix_to_camera_position(extrinsic_matrix: np.ndarray) -> np.ndarray:
    """
    :param extrinsic_matrix: 4*4 extrinsic camera matrix
    :return: 3d vector representing
    """
    inverse_rotation = np.linalg.inv(np.delete(np.delete(extrinsic_matrix, 3, 0), 3, 1))
    return inverse_rotation @ extrinsic_matrix[:3, 3]
def extrinsic_matrix_to_rotation(extrinsic_matrix: np.ndarray) -> list[float]:
    """
    :param extrinsic_matrix: 4*4 extrinsic camera matrix
    :return: the list [yaw, pitch, roll]
    """
    sy = math.sqrt(extrinsic_matrix[0, 0] * extrinsic_matrix[0, 0] + extrinsic_matrix[1, 0] * extrinsic_matrix[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(extrinsic_matrix[2, 1], extrinsic_matrix[2, 2])
        y = math.atan2(-extrinsic_matrix[2, 0], sy)
        z = math.atan2(extrinsic_matrix[1, 0], extrinsic_matrix[0, 0])
    else:
        x = math.atan2(-extrinsic_matrix[1, 2], extrinsic_matrix[1, 1])
        y = math.atan2(-extrinsic_matrix[2, 0], sy)
        z = 0

    return [x, y, z]

def draw(frame: np.ndarray, proj_tags: list[list[list[int or float]]], ids: list[int]):
    """
    :param frame: the frame on which we want to draw the tags
    :param proj_tags: a list of the projected tags coordinate on the image
    :param ids: a list of the id's of the tags
    """
    for i in range(len(ids)):
        middle = find_projected_tag_center(proj_tags[i])
        cv2.circle(frame, (int(middle[0]), int(middle[1])), 5, [0, 0, 255], 5)
        for j in range(len(proj_tags[i])):
            cv2.circle(frame, (int(proj_tags[i][j][0]), int(proj_tags[i][j][1])), 5, [255, 0, 0], 5)
            cv2.putText(frame, str(j), (int(proj_tags[i][j][0]) + 10, int(proj_tags[i][j][1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 3)


def main():
    # TODO: check if focal length values are actually good
    TAG_SIDE_LENGTH = 15.3
    TAG_DIAG_LENGTH = TAG_SIDE_LENGTH * (2**0.5)

    cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
    cam = cv2.VideoCapture(1)
    width = 1280
    height = 720
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, 30)
    diag = (width**2 + height**2)**0.5
    F_LENGTH_X_LIFECAM = math.tan(0.5355780593748425) * 2 * diag
    F_LENGTH_Y_LIFECAM = math.tan(0.3221767906849529) * 2 * diag

    while True:
        ok, frame = cam.read()


        proj_squares, ids = detect_april_tags(frame)
        for square in proj_squares:
            camera_oriented_coords = tag_projected_points_to_camera_oriented(tag=square, width=width,
                                                                        height=height, diagonal_length=TAG_DIAG_LENGTH,
                                                                        focal_length_x=F_LENGTH_X_LIFECAM,
                                                                        focal_length_y=F_LENGTH_Y_LIFECAM)
            # print(math.degrees(math.asin(0.5*((camera_oriented_coords[:3,0] + camera_oriented_coords[:3,2]) -
            #       (camera_oriented_coords[:3,1] + camera_oriented_coords[:3,3]))[2] /
            #       (0.5*(camera_oriented_coords[:3,1] + camera_oriented_coords[:3,3]))[2])))
            print(np.linalg.norm(0.25*((camera_oriented_coords[:3,0] + camera_oriented_coords[:3,2]) +
                   (camera_oriented_coords[:3,1] + camera_oriented_coords[:3,3]))))
        draw(frame, proj_squares, ids)
        cv2.imshow('Display', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()

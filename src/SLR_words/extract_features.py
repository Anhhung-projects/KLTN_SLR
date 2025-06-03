import cv2
import numpy as np
import mediapipe as mp
import os
import csv
from math import sqrt

# Khởi tạo MediaPipe Hands và Pose
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7)

def rotate_frame(frame, angle):
    """Xoay frame một góc angle (độ)."""
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))
    return rotated_frame

def normalize_landmarks(landmarks, frame_shape):
    """Chuẩn hóa điểm mốc tay theo bounding box"""
    if landmarks.sum() == 0:
        return landmarks
    landmarks = landmarks.reshape(-1, 3)
    x_coords, y_coords, z_coords = landmarks[:, 0], landmarks[:, 1], landmarks[:, 2]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    width, height = frame_shape[1], frame_shape[0]
    if x_max - x_min > 0 and y_max - y_min > 0:
        x_normalized = (x_coords - x_min) / (x_max - x_min)
        y_normalized = (y_coords - y_min) / (y_max - y_min)
    else:
        x_normalized = x_coords / width
        y_normalized = y_coords / height
    return np.concatenate([x_normalized, y_normalized, z_coords]).flatten()

def compute_hand_movement_distance(hand_landmarks, prev_center):
    """Tính khoảng cách di chuyển của trung tâm bàn tay giữa các frame."""
    if hand_landmarks.sum() == 0 or prev_center is None:
        return 0.0
    hand_landmarks = hand_landmarks.reshape(-1, 3)
    x_mean, y_mean = np.mean(hand_landmarks[:, 0]), np.mean(hand_landmarks[:, 1])
    distance = sqrt((x_mean - prev_center[0])**2 + (y_mean - prev_center[1])**2)
    normalized_distance = min(distance * 10, 1.0)
    return normalized_distance

def compute_hand_to_shoulder_distances(hand_landmarks, shoulder_left, shoulder_right, frame_shape):
    """Tính khoảng cách từ trung tâm bàn tay đến hai vai, chuẩn hóa về [0, 1]."""
    if hand_landmarks.sum() == 0:
        return 0.0, 0.0
    hand_landmarks = hand_landmarks.reshape(-1, 3)
    width, height = frame_shape[1], frame_shape[0]
    
    x_mean, y_mean = np.mean(hand_landmarks[:, 0]), np.mean(hand_landmarks[:, 1])
    center_absolute = (int(x_mean * width), int(y_mean * height))
    
    shoulder_left_absolute = shoulder_left if shoulder_left is not None else None
    shoulder_right_absolute = shoulder_right if shoulder_right is not None else None
    
    # Tính khoảng cách thô (pixel)
    dist_to_left_raw = 0.0
    dist_to_right_raw = 0.0
    
    if shoulder_left_absolute:
        dist_to_left_raw = sqrt((center_absolute[0] - shoulder_left_absolute[0])**2 + 
                                (center_absolute[1] - shoulder_left_absolute[1])**2)
    if shoulder_right_absolute:
        dist_to_right_raw = sqrt((center_absolute[0] - shoulder_right_absolute[0])**2 + 
                                 (center_absolute[1] - shoulder_right_absolute[1])**2)
    
    # Tính khoảng cách giữa hai vai
    shoulder_distance = 0.0
    if shoulder_left_absolute and shoulder_right_absolute:
        shoulder_distance = sqrt((shoulder_left_absolute[0] - shoulder_right_absolute[0])**2 + 
                                 (shoulder_left_absolute[1] - shoulder_right_absolute[1])**2)
    
    # xử lý ngoại lệ
    diagonal = sqrt(width**2 + height**2)
    normalization_factor = shoulder_distance if shoulder_distance > 0 else diagonal
    
    # Chuẩn hóa khoảng cách về [0, 1]
    dist_to_left = dist_to_left_raw / normalization_factor if normalization_factor > 0 else 0.0
    dist_to_right = dist_to_right_raw / normalization_factor if normalization_factor > 0 else 0.0
    
    return dist_to_left, dist_to_right

def extract_features(video_path, output_dir, metadata, video_id, label, video_type, rotation_angle=0):
    """Trích xuất đặc trưng từ video cho cả hai tay, lưu vào file .npy và cập nhật metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    features = []
    frame_data = []
    right_features_list = []
    left_features_list = []
    prev_right = None
    prev_left = None
    prev_right_center = None
    prev_left_center = None
    prev_right_shoulder_dists = (0.0, 0.0)
    prev_left_shoulder_dists = (0.0, 0.0)
    
    # biến xử lý ngoại lệ
    right_temp_count = 0
    left_temp_count = 0
    right_temp_start = -1
    left_temp_start = -1
    max_temp_frames = 5
    prev_shoulder_left = None
    prev_shoulder_right = None
    frame_idx = 0
    
    while cap.isOpened() and len(frame_data) < 30:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 2 == 0:
            if rotation_angle != 0:
                frame = rotate_frame(frame, rotation_angle)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_shape = frame.shape
            
            # Nhận diện tay
            hand_results = hands.process(frame_rgb)
            right_hand, left_hand = None, None
            right_score, left_score = 0, 0
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hand, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    score = handedness.classification[0].score
                    hand_label = handedness.classification[0].label
                    if hand_label == 'Right' and (right_hand is None or score > right_score):
                        right_hand = hand
                        right_score = score
                    elif hand_label == 'Left' and (left_hand is None or score > left_score):
                        left_hand = hand
                        left_score = score
            
            # Nhận diện vai
            pose_results = pose.process(frame_rgb)
            shoulder_left, shoulder_right = None, None
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                shoulder_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_shape[1]),
                               int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_shape[0]))
                shoulder_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_shape[1]),
                                int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_shape[0]))
                prev_shoulder_left = shoulder_left
                prev_shoulder_right = shoulder_right
            else:
                shoulder_left = prev_shoulder_left
                shoulder_right = prev_shoulder_right
            
            # Trích xuất đặc trưng tay phải
            right_features = np.zeros(21 * 3) if not right_hand else np.array([
                coord for lm in right_hand.landmark for coord in [lm.x, lm.y, lm.z]
            ])
            right_detected = right_features.sum() != 0
            is_right_temp = False
            if not right_detected and prev_right is not None:
                right_features = prev_right
                is_right_temp = True
                if right_temp_start == -1:
                    right_temp_start = len(features)
                right_temp_count += 1
            else:
                right_temp_count = 0
                right_temp_start = -1
            
            right_features_normalized = normalize_landmarks(right_features, frame_shape)
            right_movement_distance = compute_hand_movement_distance(right_features, prev_right_center)
            right_shoulder_dists = compute_hand_to_shoulder_distances(right_features, shoulder_left, shoulder_right, frame_shape)
            
            # Trích xuất đặc trưng tay trái
            left_features = np.zeros(21 * 3) if not left_hand else np.array([
                coord for lm in left_hand.landmark for coord in [lm.x, lm.y, lm.z]
            ])
            left_detected = left_features.sum() != 0
            is_left_temp = False
            if not left_detected and prev_left is not None:
                left_features = prev_left
                is_left_temp = True
                if left_temp_start == -1:
                    left_temp_start = len(features)
                left_temp_count += 1
            else:
                left_temp_count = 0
                left_temp_start = -1
            
            left_features_normalized = normalize_landmarks(left_features, frame_shape)
            left_movement_distance = compute_hand_movement_distance(left_features, prev_left_center)
            left_shoulder_dists = compute_hand_to_shoulder_distances(left_features, shoulder_left, shoulder_right, frame_shape)
            
            # Kiểm tra và bỏ đặc trưng tái sử dụng nếu vượt quá ngưỡng
            if right_temp_count >= max_temp_frames and right_temp_start != -1:
                for i in range(right_temp_start, len(features)):
                    features[i][:65] = np.zeros(65)  # Bỏ đặc trưng tay phải
                prev_right = None
                prev_right_center = None
                prev_right_shoulder_dists = (0.0, 0.0)
                right_temp_count = 0
                right_temp_start = -1
            
            if left_temp_count >= max_temp_frames and left_temp_start != -1:
                for i in range(left_temp_start, len(features)):
                    features[i][65:] = np.zeros(67)  # Bỏ đặc trưng tay trái
                prev_left = None
                prev_left_center = None
                prev_left_shoulder_dists = (0.0, 0.0)
                left_temp_count = 0
                left_temp_start = -1
            
            # Kết hợp đặc trưng
            frame_features = np.concatenate([
                right_features_normalized, [right_movement_distance],
                [prev_right_shoulder_dists[0] if shoulder_left is None else right_shoulder_dists[0],
                 prev_right_shoulder_dists[1] if shoulder_right is None else right_shoulder_dists[1]],
                left_features_normalized, [left_movement_distance],
                [prev_left_shoulder_dists[0] if shoulder_left is None else left_shoulder_dists[0],
                 prev_left_shoulder_dists[1] if shoulder_right is None else left_shoulder_dists[1]]
            ])
            features.append(frame_features)
            
            # Cập nhật trạng thái nếu tay được phát hiện
            if right_detected:
                right_landmarks = right_features.reshape(-1, 3)
                prev_right_center = (np.mean(right_landmarks[:, 0]), np.mean(right_landmarks[:, 1]))
                prev_right = right_features
                prev_right_shoulder_dists = right_shoulder_dists
            if left_detected:
                left_landmarks = left_features.reshape(-1, 3)
                prev_left_center = (np.mean(left_landmarks[:, 0]), np.mean(left_landmarks[:, 1]))
                prev_left = left_features
                prev_left_shoulder_dists = left_shoulder_dists
            
            frame_data.append(frame_idx)
            right_features_list.append(right_features)
            left_features_list.append(left_features)
        
        frame_idx += 1
    
    # Lặp frame cuối nếu không đủ 30
    while len(features) < 30:
        features.append(features[-1] if features else np.zeros(132))
        frame_data.append(frame_data[-1] if frame_data else 0)
        right_features_list.append(right_features_list[-1] if right_features_list else np.zeros(21 * 3))
        left_features_list.append(left_features_list[-1] if left_features_list else np.zeros(21 * 3))
    
    cap.release()
    
    features = np.array(features)
    feature_path = os.path.join(output_dir, f"video_{video_id}_{video_type}.npy")
    np.save(feature_path, features)
    
    metadata.append({
        'video_id': video_id,
        'label': label,
        'type': video_type,
        'feature_path': feature_path
    })

def process_all_videos(data_dir):
    """Duyệt tất cả ký hiệu và video, trích xuất đặc trưng."""
    output_dir = os.path.join(os.path.dirname(data_dir), 'word_features')
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = []
    video_id = 1
    
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir) or label in ['features', 'hand_checking']:
            continue
        
        for video_name in sorted(os.listdir(label_dir)):
            if not video_name.endswith('.avi'):
                continue
            video_path = os.path.join(label_dir, video_name)
            
            print(f"Hand feature extracting: {video_path}")
            extract_features(video_path, output_dir, metadata, video_id, label, 'original', rotation_angle=0)
            
            print(f"Hand feature extracting: {video_path} (rotated_15)")
            extract_features(video_path, output_dir, metadata, video_id, label, 'rotated_15', rotation_angle=15)
            
            print(f"Hand feature extracting: {video_path} (rotated_-15)")
            extract_features(video_path, output_dir, metadata, video_id, label, 'rotated_-15', rotation_angle=-15)
            
            video_id += 1
    
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'label', 'type', 'feature_path'])
        writer.writeheader()
        for row in metadata:
            writer.writerow(row)

data_dir = './data/words'
process_all_videos(data_dir)

hands.close()
pose.close()
import cv2
import numpy as np
import mediapipe as mp
import os
import csv
from math import sqrt

DATA_DIR = './data/alphabet'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, frame_rgb

def rotate_frame(frame, angle):
    #xoay nhẹ khung hình
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))
    return rotated_frame

def normalize_landmarks(landmarks, frame_shape):
    #chuẩn hóa điểm mốc
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
    z_normalized = z_coords * 0.5 #giảm trọng số z
    return np.concatenate([x_normalized, y_normalized, z_normalized]).flatten()

def compute_hand_movement_distance(hand_landmarks, prev_center):
    if hand_landmarks.sum() == 0 or prev_center is None:
        return 0.0
    hand_landmarks = hand_landmarks.reshape(-1, 3)
    x_mean, y_mean = np.mean(hand_landmarks[:, 0]), np.mean(hand_landmarks[:, 1])
    distance = sqrt((x_mean - prev_center[0])**2 + (y_mean - prev_center[1])**2) #tính khoảng cách
    normalized_distance = min(distance * 10, 1.0)  # Phóng đại x10, giới hạn [0, 1]
    return normalized_distance

def extract_features(video_path, output_dir, metadata, video_id, label, video_type, rotation_angle=0):
    """Trích xuất đặc trưng từ video, lưu vào file .npy và cập nhật metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    features = []
    frame_data = []
    right_features_list = []
    prev_right = None
    prev_center = None
    frame_idx = 0
    
    while cap.isOpened() and len(frame_data) < 30:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 2 == 0:  # 1 skip 1
            if rotation_angle != 0:
                frame = rotate_frame(frame, rotation_angle)
            _, frame_rgb = preprocess_frame(frame)
            frame_shape = frame.shape
            
            hand_results = hands.process(frame_rgb)
            right_hand, right_score = None, 0
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hand, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    if handedness.classification[0].label == 'Right':
                        score = handedness.classification[0].score
                        if right_hand is None or score > right_score:
                            right_hand = hand
                            right_score = score
            
            right_features = np.zeros(21 * 3) if not right_hand else np.array([
                coord for lm in right_hand.landmark for coord in [lm.x, lm.y, lm.z]
            ])
            
            if right_features.sum() == 0 and prev_right is not None:
                right_features = prev_right
            
            right_features_normalized = normalize_landmarks(right_features, frame_shape)
            hand_movement_distance = compute_hand_movement_distance(right_features, prev_center)
            
            if right_features.sum() != 0:
                right_landmarks = right_features.reshape(-1, 3)
                prev_center = (np.mean(right_landmarks[:, 0]), np.mean(right_landmarks[:, 1]))
                prev_right = right_features
            
            frame_features = np.concatenate([right_features_normalized, [hand_movement_distance]])
            features.append(frame_features)
            
            frame_data.append(frame_idx)
            right_features_list.append(right_features)
        
        frame_idx += 1
    
    #lặp frame cuối nếu không đủ 30
    while len(features) < 30:
        features.append(features[-1] if features else np.zeros(64))
        frame_data.append(frame_data[-1] if frame_data else 0)
        right_features_list.append(right_features_list[-1] if right_features_list else np.zeros(21 * 3))
    
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
    output_dir = os.path.join(os.path.dirname(data_dir), 'features')
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

#main
process_all_videos(DATA_DIR)
hands.close()
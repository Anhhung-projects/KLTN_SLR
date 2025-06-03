import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from collections import deque
import pandas as pd
import os
from math import sqrt
import uuid

THRESHOLD = 0.8
CAM_IDX = 0  # Thay đổi nếu kết nối với webcam

# Khởi tạo MediaPipe Hands và Pose
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7)

# Đọc nhãn từ metadata.csv
METADATA_PATH = './data/word_features/metadata.csv'
metadata = pd.read_csv(METADATA_PATH)
labels = sorted(metadata['label'].unique())
label_encoder = LabelEncoder()
label_encoder.fit(labels)
print(f"Loaded {len(labels)} labels: {labels}")

# Load model
model = tf.keras.models.load_model('./model/SLR_model_words.h5', compile=False)

# Tiền xử lý frame
def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, frame_rgb

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
    """Tính khoảng cách di chuyển của trung tâm bàn tay giữa các frame"""
    if hand_landmarks.sum() == 0 or prev_center is None:
        return 0.0
    hand_landmarks = hand_landmarks.reshape(-1, 3)
    x_mean, y_mean = np.mean(hand_landmarks[:, 0]), np.mean(hand_landmarks[:, 1])
    distance = sqrt((x_mean - prev_center[0])**2 + (y_mean - prev_center[1])**2)
    normalized_distance = min(distance * 10, 1.0)
    return normalized_distance

def compute_hand_to_shoulder_distances(hand_landmarks, shoulder_left, shoulder_right, frame_shape):
    """Tính khoảng cách từ trung tâm bàn tay đến hai vai, chuẩn hóa về [0, 1]"""
    if hand_landmarks.sum() == 0:
        return 0.0, 0.0
    hand_landmarks = hand_landmarks.reshape(-1, 3)
    width, height = frame_shape[1], frame_shape[0]
    
    x_mean, y_mean = np.mean(hand_landmarks[:, 0]), np.mean(hand_landmarks[:, 1])
    center_absolute = (int(x_mean * width), int(y_mean * height))
    
    shoulder_left_absolute = (int(shoulder_left[0] * width), int(shoulder_left[1] * height)) if shoulder_left is not None else None
    shoulder_right_absolute = (int(shoulder_right[0] * width), int(shoulder_right[1] * height)) if shoulder_right is not None else None
    
    dist_to_left_raw = 0.0
    dist_to_right_raw = 0.0
    
    if shoulder_left_absolute:
        dist_to_left_raw = sqrt((center_absolute[0] - shoulder_left_absolute[0])**2 + 
                                (center_absolute[1] - shoulder_left_absolute[1])**2)
    if shoulder_right_absolute:
        dist_to_right_raw = sqrt((center_absolute[0] - shoulder_right_absolute[0])**2 + 
                                 (center_absolute[1] - shoulder_right_absolute[1])**2)
    
    shoulder_distance = 0.0
    if shoulder_left_absolute and shoulder_right_absolute:
        shoulder_distance = sqrt((shoulder_left_absolute[0] - shoulder_right_absolute[0])**2 + 
                                 (shoulder_left_absolute[1] - shoulder_right_absolute[1])**2)
    
    diagonal = sqrt(width**2 + height**2)
    normalization_factor = shoulder_distance if shoulder_distance > 0 else diagonal
    
    dist_to_left = dist_to_left_raw / normalization_factor if normalization_factor > 0 else 0.0
    dist_to_right = dist_to_right_raw / normalization_factor if normalization_factor > 0 else 0.0
    
    return dist_to_left, dist_to_right

def extract_frame_features(frame, prev_right, prev_left, prev_right_center, prev_left_center, 
                         prev_right_shoulder_dists, prev_left_shoulder_dists, prev_shoulder_left, prev_shoulder_right):
    """Trích xuất đặc trưng từ frame"""
    processed_frame, frame_rgb = preprocess_frame(frame)
    frame_shape = frame.shape
    
    hand_results = hands.process(frame_rgb)
    right_hand, left_hand = None, None
    right_score, left_score = 0, 0
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            score = handedness.classification[0].score
            label_hand = handedness.classification[0].label
            if label_hand == 'Right' and (right_hand is None or score > right_score):
                right_hand = hand
                right_score = score
            elif label_hand == 'Left' and (left_hand is None or score > left_score):
                left_hand = hand
                left_score = score
    
    pose_results = pose.process(frame_rgb)
    shoulder_left, shoulder_right = None, None
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        shoulder_left = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
        shoulder_right = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
    else:
        shoulder_left = prev_shoulder_left
        shoulder_right = prev_shoulder_right
    
    # Trích xuất đặc trưng tay phải
    right_features = np.zeros(21 * 3) if not right_hand else np.array([
        coord for lm in right_hand.landmark for coord in [lm.x, lm.y, lm.z]
    ])
    right_features_normalized = normalize_landmarks(right_features, frame_shape)
    right_movement_distance = compute_hand_movement_distance(right_features, prev_right_center)
    right_shoulder_dists = compute_hand_to_shoulder_distances(right_features, shoulder_left, shoulder_right, frame_shape)
    right_detected = right_features.sum() != 0
    if right_detected:
        right_landmarks = right_features.reshape(-1, 3)
        prev_right_center = (np.mean(right_landmarks[:, 0]), np.mean(right_landmarks[:, 1]))
        prev_right = right_features
        prev_right_shoulder_dists = right_shoulder_dists
    
    # Trích xuất đặc trưng tay trái
    left_features = np.zeros(21 * 3) if not left_hand else np.array([
        coord for lm in left_hand.landmark for coord in [lm.x, lm.y, lm.z]
    ])
    left_features_normalized = normalize_landmarks(left_features, frame_shape)
    left_movement_distance = compute_hand_movement_distance(left_features, prev_left_center)
    left_shoulder_dists = compute_hand_to_shoulder_distances(left_features, shoulder_left, shoulder_right, frame_shape)
    left_detected = left_features.sum() != 0
    if left_detected:
        left_landmarks = left_features.reshape(-1, 3)
        prev_left_center = (np.mean(left_landmarks[:, 0]), np.mean(left_landmarks[:, 1]))
        prev_left = left_features
        prev_left_shoulder_dists = left_shoulder_dists
    
    # Kết hợp đặc trưng
    frame_features = np.concatenate([
        right_features_normalized, [right_movement_distance],
        [prev_right_shoulder_dists[0] if shoulder_left is None else right_shoulder_dists[0],
         prev_right_shoulder_dists[1] if shoulder_right is None else right_shoulder_dists[1]],
        left_features_normalized, [left_movement_distance],
        [prev_left_shoulder_dists[0] if shoulder_left is None else left_shoulder_dists[0],
         prev_left_shoulder_dists[1] if shoulder_right is None else left_shoulder_dists[1]]
    ])
    
    hand_detected = right_detected or left_detected
    return (frame_features, hand_detected, prev_right, prev_left, 
            prev_right_center, prev_left_center, right_shoulder_dists, left_shoulder_dists,
            shoulder_left, shoulder_right)

def display_progress_bar(frame, current_frames, total_frames=30):
    """Hiển thị thanh tiến trình"""
    h, w, _ = frame.shape
    bar_width = 200
    bar_height = 10  
    bar_x = (w - bar_width) // 2
    bar_y = h - 30
    filled_width = int(bar_width * current_frames / total_frames)
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 2)
    
    return frame

def display_predictions(frame_full, predictions):
    """Hiển thị từ/cụm từ nhận dạng"""
    h, w, _ = frame_full.shape
    sidebar_width = w // 4  
    font_scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    y_offset = 50
    line_spacing = 25
    
    cv2.rectangle(frame_full, (w - sidebar_width, 0), (w, h), (50, 50, 50), -1)
    
    for i, pred in enumerate(predictions):
        if i >= 7:
            break
        y_pos = y_offset + i * line_spacing
        cv2.putText(frame_full, pred, (w - sidebar_width + 10, y_pos), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return frame_full

def real_time_sign_recognition():
    """Triển khai mô hình SLR"""
    cap = cv2.VideoCapture(CAM_IDX)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        cap.release()
        return
    h, w = frame.shape[:2]
    
    sidebar_width = w // 4
    new_w = w + sidebar_width 
    new_h = h  
    video_width = w
    
    window = deque(maxlen=30)
    prev_right = None
    prev_left = None
    prev_right_center = None
    prev_left_center = None
    prev_right_shoulder_dists = (0.0, 0.0)
    prev_left_shoulder_dists = (0.0, 0.0)
    prev_shoulder_left = None
    prev_shoulder_right = None
    right_temp_count = 0
    left_temp_count = 0
    right_temp_start = -1
    left_temp_start = -1
    max_temp_frames = 5
    no_hand_count = 0
    frame_idx = 0
    predictions = deque(maxlen=7)
    
    print("Hand detecting...")
    hand_detecting_logged = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (video_width, new_h))
        
        frame_full = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        frame_full[:, :video_width] = frame
        
        frame_idx += 1
        if frame_idx % 2 != 0:
            frame_full = display_progress_bar(frame_full, len(window))
            frame_full = display_predictions(frame_full, predictions)
            cv2.imshow('Sign Language Recognition', frame_full)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Trích xuất đặc trưng
        (frame_features, hand_detected, prev_right, prev_left, 
         prev_right_center, prev_left_center, right_shoulder_dists, 
         left_shoulder_dists, shoulder_left, shoulder_right) = extract_frame_features(
            frame, prev_right, prev_left, prev_right_center, prev_left_center,
            prev_right_shoulder_dists, prev_left_shoulder_dists, prev_shoulder_left, prev_shoulder_right
        )
        
        is_right_temp = False
        if not (frame_features[:63].sum() != 0) and prev_right is not None:
            frame_features[:63] = normalize_landmarks(prev_right, frame.shape)
            frame_features[63] = 0.0
            frame_features[64:66] = prev_right_shoulder_dists
            is_right_temp = True
            if right_temp_start == -1:
                right_temp_start = len(window)
            right_temp_count += 1
        else:
            right_temp_count = 0
            right_temp_start = -1
        
        is_left_temp = False
        if not (frame_features[66:129].sum() != 0) and prev_left is not None:
            frame_features[66:129] = normalize_landmarks(prev_left, frame.shape)
            frame_features[129] = 0.0
            frame_features[130:132] = prev_left_shoulder_dists
            is_left_temp = True
            if left_temp_start == -1:
                left_temp_start = len(window)
            left_temp_count += 1
        else:
            left_temp_count = 0
            left_temp_start = -1
        
        if right_temp_count >= max_temp_frames and right_temp_start != -1:
            for i in range(right_temp_start, len(window)):
                window[i][:65] = np.zeros(65)
            prev_right = None
            prev_right_center = None
            prev_right_shoulder_dists = (0.0, 0.0)
            right_temp_count = 0
            right_temp_start = -1
        
        if left_temp_count >= max_temp_frames and left_temp_start != -1:
            for i in range(left_temp_start, len(window)):
                window[i][65:] = np.zeros(67)
            prev_left = None
            prev_left_center = None
            prev_left_shoulder_dists = (0.0, 0.0)
            left_temp_count = 0
            left_temp_start = -1
        
        if len(window) == 0 and not hand_detected:
            if not hand_detecting_logged:
                print("Hand detecting...")
                hand_detecting_logged = True
            frame_full = display_progress_bar(frame_full, len(window))
            frame_full = display_predictions(frame_full, predictions)
            cv2.imshow('Sign Language Recognition', frame_full)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        if not hand_detected:
            no_hand_count += 1
        else:
            no_hand_count = 0
            if hand_detecting_logged:
                hand_detecting_logged = False
        
        if no_hand_count >= 10:
            print("Lost hand")
            print("Hand detecting...")
            window.clear()
            prev_right = None
            prev_left = None
            prev_right_center = None
            prev_left_center = None
            prev_right_shoulder_dists = (0.0, 0.0)
            prev_left_shoulder_dists = (0.0, 0.0)
            prev_shoulder_left = None
            prev_shoulder_right = None
            no_hand_count = 0
            right_temp_count = 0
            left_temp_count = 0
            right_temp_start = -1
            left_temp_start = -1
            frame_full = display_progress_bar(frame_full, len(window))
            frame_full = display_predictions(frame_full, predictions)
            cv2.imshow('Sign Language Recognition', frame_full)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        window.append(frame_features)
        print(f"Collecting: {len(window)}/30")
        
        if len(window) == 30:
            print("Recognizing...")
            features = np.array(window)
            features = features[np.newaxis, ...]  # (1, 30, 132)
            pred = model.predict(features, verbose=0)
            confidence = np.max(pred)
            if confidence >= THRESHOLD:
                pred_class = np.argmax(pred, axis=1)
                predicted_label = label_encoder.inverse_transform(pred_class)[0]
                predictions.append(predicted_label)
                print(f"Predicted: {predicted_label} (confidence: {confidence:.2f})")
            else:
                print(f"Low confidence: {confidence:.2f}, prediction skipped")
            
            window.clear()
            prev_right = None
            prev_left = None
            prev_right_center = None
            prev_left_center = None
            prev_right_shoulder_dists = (0.0, 0.0)
            prev_left_shoulder_dists = (0.0, 0.0)
            prev_shoulder_left = None
            prev_shoulder_right = None
            no_hand_count = 0
            right_temp_count = 0
            left_temp_count = 0
            right_temp_start = -1
            left_temp_start = -1
        
        frame_full = display_progress_bar(frame_full, len(window))
        frame_full = display_predictions(frame_full, predictions)
        cv2.imshow('Sign Language Recognition', frame_full)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()

if __name__ == "__main__":
    real_time_sign_recognition()
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from collections import deque
import time
from math import sqrt
import json
from PIL import Image, ImageDraw, ImageFont

THRESHOLD = 0.8
CAM_IDX = 1

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

with open('./data/label_mapping_alphabet.json', 'r', encoding='utf-8') as f:
    label_mapping = json.load(f)

model = tf.keras.models.load_model('./model/SLR_model_alphabet.h5', compile=False) #load model
labels = list(label_mapping.keys()) #["A", "AW", "AA", ...]
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Hàm tiền xử lý frame
# def preprocess_frame(frame):
#     frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
#     frame = cv2.GaussianBlur(frame, (5, 5), 0)
#     blurred = cv2.GaussianBlur(frame, (0, 0), 3)
#     frame = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     return frame, frame_rgb

def normalize_landmarks(landmarks, frame_shape):
    """Chuẩn hóa điểm mốc bàn tay"""
    if landmarks.sum() == 0: #không phát hiện bàn tay
        return landmarks 
    landmarks = landmarks.reshape(-1, 3)
    x_coords, y_coords, z_coords = landmarks[:, 0], landmarks[:, 1], landmarks[:, 2]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    width, height = frame_shape[1], frame_shape[0]
    if x_max - x_min > 0 and y_max - y_min > 0:
        x_normalized = (x_coords - x_min) / (x_max - x_min)
        y_normalized = (y_coords - y_min) / (y_max - y_min) #x,y chuẩn hóa theo min-max (bounding box)
    else:
        x_normalized = x_coords / width
        y_normalized = y_coords / height
    z_normalized = z_coords * 0.5 #giảm ảnh hưởng của z
    return np.concatenate([x_normalized, y_normalized, z_normalized]).flatten() #shape (63,)

def compute_hand_movement_distance(hand_landmarks, prev_center):
    """tính toán khoảng di chuyển bàn tay so với khung hình trước"""
    if hand_landmarks.sum() == 0 or prev_center is None:
        return 0.0
    hand_landmarks = hand_landmarks.reshape(-1, 3)
    x_mean, y_mean = np.mean(hand_landmarks[:, 0]), np.mean(hand_landmarks[:, 1]) #tính điểm giữa bàn tay
    distance = sqrt((x_mean - prev_center[0])**2 + (y_mean - prev_center[1])**2) #khoảng cách Euclidean
    normalized_distance = min(distance * 10, 1.0) #chuẩn hóa
    return normalized_distance

def extract_frame_features(frame, prev_right, prev_center):
    """Hàm trích xuất đặc trưng từ frame"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    hand_detected = right_features.sum() != 0
    if not hand_detected and prev_right is not None:
        right_features = prev_right

    right_features_normalized = normalize_landmarks(right_features, frame_shape)
    hand_movement_distance = compute_hand_movement_distance(right_features, prev_center)

    if hand_detected:
        right_landmarks = right_features.reshape(-1, 3)
        prev_center = (np.mean(right_landmarks[:, 0]), np.mean(right_landmarks[:, 1]))
        prev_right = right_features
    else:
        prev_center = prev_center if prev_center is not None else None
        prev_right = prev_right if prev_right is not None else None

    frame_features = np.concatenate([right_features_normalized, [hand_movement_distance]])
    return frame_features, hand_detected, prev_right, prev_center

def display_subtitles(frame, predictions, word_sequences, window_len, max_frames=30):
    """Hàm hiển thị phụ đề và chuỗi ký hiệu bằng Pillow"""
    h, w, _ = frame.shape
    
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    
    try:
        font_path = "./font/arial.ttf"  # font hỗ trợ tiếng Việt
        font = ImageFont.truetype(font_path, 30)  
        font_small = ImageFont.truetype(font_path, 20)  
    except:
        # Nếu không tìm thấy Arial, sử dụng font mặc định của Pillow
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Ánh xạ predictions và word_sequences sang ký tự tiếng Việt
    mapped_predictions = [label_mapping.get(pred, pred) for pred in predictions]
    mapped_word_sequences = [''.join(label_mapping.get(c, c) for c in word) for word in word_sequences]

    text = " ".join(mapped_predictions)

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]  # right - left
    text_height = text_bbox[3] - text_bbox[1]  # bottom - top
    text_x = (w - text_width) // 2
    text_y = h - 40

    draw.rectangle((text_x - 10, text_y - 30, text_x + text_width + 10, text_y + text_height + 10), fill=(0, 0, 0))
    draw.text((text_x, text_y - 20), text, font=font, fill=(255, 255, 255))

    # thanh tiến trình
    progress = window_len / max_frames
    bar_x = (w - 80) // 2
    bar_y = h - 20
    bar_width = 100
    bar_height = 5
    
    draw.rectangle((bar_x, bar_y, bar_x + bar_width, bar_y + bar_height), fill=(0, 0, 0))
    draw.rectangle((bar_x, bar_y, bar_x + int(bar_width * progress), bar_y + bar_height), fill=(0, 255, 0))
    draw.rectangle((bar_x, bar_y, bar_x + bar_width, bar_y + bar_height), outline=(255, 255, 255), width=1)

    max_text_width = 0
    for word in mapped_word_sequences:
        word_bbox = draw.textbbox((0, 0), word, font=font_small)
        word_width = word_bbox[2] - word_bbox[0]
        max_text_width = max(max_text_width, word_width)

    if mapped_word_sequences:
        draw.rectangle((0, 0, max_text_width + 20, 10 + len(mapped_word_sequences) * 30), fill=(0, 0, 0))

    for i, word in enumerate(mapped_word_sequences):
        draw.text((10, 10 + i * 30), word, font=font_small, fill=(255, 255, 255))

    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    return frame

def real_time_sign_recognition():
    """Nhận dạng ngôn ngữ ký hiệu thời gian thực"""
    cap = cv2.VideoCapture(CAM_IDX)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read initial frame")
        return
    h, w = frame.shape[:2]
    new_width = int(w * 1.5)
    new_height = int(h * 1.5)

    cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sign Language Recognition', new_width, new_height)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    window = deque(maxlen=30)
    predictions = deque(maxlen=7)
    word_sequences = deque(maxlen=5)
    prev_right = None
    prev_center = None
    no_hand_count = 0
    frame_idx = 0
    is_predicting = False
    last_prediction_time = 0
    last_hand_time = time.time()
    current_word = []
    hand_detecting_start_time = None

    print("Hand detecting...")
    hand_detecting_logged = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        frame = cv2.flip(frame, 1)
        frame_idx += 1
        if frame_idx % 2 != 0:
            cv2.imshow('Sign Language Recognition', display_subtitles(frame, predictions, word_sequences, len(window)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        frame_features, hand_detected, prev_right, prev_center = extract_frame_features(frame, prev_right, prev_center)

        if hand_detected:
            last_hand_time = time.time()
            hand_detecting_start_time = None
        else:
            if hand_detecting_start_time is None:
                hand_detecting_start_time = time.time()

        if len(window) == 0 and not hand_detected:
            if not hand_detecting_logged:
                print("Hand detecting...")
                hand_detecting_logged = True
            if hand_detecting_start_time and time.time() - hand_detecting_start_time >= 2 and current_word:
                # Ánh xạ current_word sang ký tự tiếng Việt trước khi tạo word
                mapped_word = ''.join(label_mapping.get(c, c) for c in current_word)
                word_sequences.append(mapped_word)
                current_word = []
                predictions.clear()
                print("Reset...")
            cv2.imshow('Sign Language Recognition', display_subtitles(frame, predictions, word_sequences, len(window)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if not hand_detected:
            no_hand_count += 1
        else:
            no_hand_count = 0
            if hand_detecting_logged:
                hand_detecting_logged = False

        if no_hand_count >= 5 or (time.time() - last_hand_time >= 2 and current_word):
            if current_word:
                mapped_word = ''.join(label_mapping.get(c, c) for c in current_word)
                word_sequences.append(mapped_word)
                current_word = []
                predictions.clear()
            print("Lost hand")
            print("Hand detecting...")
            window.clear()
            prev_right = None
            prev_center = None
            no_hand_count = 0
            is_predicting = False
            hand_detecting_start_time = time.time()
            cv2.imshow('Sign Language Recognition', display_subtitles(frame, predictions, word_sequences, len(window)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if not is_predicting:
            window.append(frame_features)
            print(f"Collecting: {len(window)}/30")

        if len(window) == 30 and not is_predicting:
            is_predicting = True
            print("Recognizing...")
            features = np.array(window)
            features = features[np.newaxis, ...]
            pred = model.predict(features, verbose=0)
            confidence = np.max(pred)
            if confidence >= THRESHOLD:
                pred_class = np.argmax(pred, axis=1)
                predicted_label = label_encoder.inverse_transform(pred_class)[0]
                predictions.append(predicted_label)
                current_word.append(predicted_label)
                print(f"Predicted: {label_mapping.get(predicted_label, predicted_label)} (confidence: {confidence:.2f})")
            else:
                print(f"Low confidence: {confidence:.2f}, prediction skipped")

            window.clear()
            prev_right = None
            prev_center = None
            no_hand_count = 0
            last_prediction_time = time.time()

        if is_predicting and time.time() - last_prediction_time >= 0.5:
            is_predicting = False

        frame = display_subtitles(frame, predictions, word_sequences, len(window))
        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    real_time_sign_recognition()
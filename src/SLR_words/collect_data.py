import os
import cv2
import time
import mediapipe as mp

NUM_OF_VIDEOS = 30
DATA_DIR = './data/words'
SIZE = (640, 480)
CAM_IDX = 1 #webcam

mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils

def draw_mediapipe_landmarks(frame, hand_results, class_name=None, video_count=None, total_videos=None):
    annotated_frame = frame.copy()
    
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            is_right_hand = handedness.classification[0].label == 'Right'
            point_color = (0, 255, 0) if is_right_hand else (0, 0, 255) #tay phải: xanh lá & đỏ
            line_color = (255, 0, 0) if is_right_hand else (255, 255, 0) #tay trái: xanh dương và vàng
            
            mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=point_color, thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=line_color, thickness=1)
            )

    if class_name:
        # hiển thị tên lớp
        text = f"Class: {class_name}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        bg_top_left = (15, 10)
        bg_bottom_right = (15 + text_size[0] + 10, 10 + text_size[1] + 10)
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, bg_top_left, bg_bottom_right, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
        cv2.putText(annotated_frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    
    if video_count is not None and total_videos is not None:
        #hiển thị số video thu thập được
        text = f"Video: {video_count}/{total_videos}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        bg_top_left = (15, 40)
        bg_bottom_right = (15 + text_size[0] + 10, 40 + text_size[1] + 10)
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, bg_top_left, bg_bottom_right, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
        cv2.putText(annotated_frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    return annotated_frame

def collect_class_data(class_name, num_of_videos=NUM_OF_VIDEOS, datadir=DATA_DIR):
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

    class_dir = os.path.join(datadir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    cap = cv2.VideoCapture(CAM_IDX)
    if not cap.isOpened():
        print("Can't open camera!")
        return False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera. Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            return False

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        hand_results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        display_frame = draw_mediapipe_landmarks(frame, hand_results, class_name=class_name)
        cv2.imshow('Camera', display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break

    collected = 0
    while collected < num_of_videos:
        for countdown in range(4, -1, -1):
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera. Exiting.")
                cap.release()
                cv2.destroyAllWindows()
                return False
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            hand_results = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            display_frame = draw_mediapipe_landmarks(frame, hand_results, class_name=class_name,
                                                  video_count=collected+1, total_videos=num_of_videos)
            cv2.putText(display_frame, f"Waitting {(1/5) * countdown if countdown != 0 else 0:.1f}s", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Chữ xanh lá, font 1
            cv2.imshow('Camera', display_frame)
            cv2.waitKey(200)

        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        filename = os.path.join(class_dir, f"video_{collected+1}.avi")
        out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))

        start_time = time.time()
        while time.time() - start_time < 3:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera. Exiting.")
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                return False
            
            frame = cv2.flip(frame, 1)
            out.write(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            hand_results = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            display_frame = draw_mediapipe_landmarks(frame, hand_results, class_name=class_name,
                                                  video_count=collected+1, total_videos=num_of_videos)
            cv2.putText(display_frame, f"R: {time.time() - start_time:.1f}s", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Chữ đỏ, font 1
            cv2.imshow('Camera', display_frame)
            cv2.waitKey(1)

        out.release()
        collected += 1

    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print(f"Finished collecting for class '{class_name}'.")
    return True

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    while True:
        class_name = input("Enter class name: ").strip()
        success = collect_class_data(class_name)
        if not success:
            break

        while True:
            continue_collection = input("Continue collecting data? (y/n): ").strip().lower()
            if continue_collection in ['y', 'n']:
                break
            print("Please enter 'y' or 'n'.")
        if continue_collection == 'n':
            break

    print("All data collection sessions complete.")

if __name__ == "__main__":
    main()
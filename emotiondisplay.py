import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

corner_ratio = (random.randint(1,5) / 10)

k1 = ''
k2 = ''
k3 = ''
k4 = ''

def change(keyword1,keyword2,keyword3,keyword4):
    k1 = keyword1
    k2 = keyword2
    k3 = keyword3
    k4 = keyword4


def get_corner_positions(frame,ratio):
  rows, cols, channels = frame.shape
  corner_size = int(min(rows, cols) * ratio)

  top_left = {
      "x": 0,
      "y": 0,
      "width": corner_size,
      "height": corner_size
  }
  top_right = {
      "x": cols - corner_size,
      "y": 0,
      "width": corner_size,
      "height": corner_size
  }
  bottom_left = {
      "x": 0,
      "y": rows - corner_size,
      "width": corner_size,
      "height": corner_size
  }
  bottom_right = {
      "x": cols - corner_size,
      "y": rows - corner_size,
      "width": corner_size,
      "height": corner_size
  }
  return top_left, top_right, bottom_left, bottom_right

def place_image_on_corner(frame, image_path, corner_position):
  corner_x = corner_position["x"]
  corner_y = corner_position["y"]
  corner_width = corner_position["width"]
  corner_height = corner_position["height"]

  corner_image = Image.open(image_path)

  corner_image = corner_image.resize((corner_width, corner_height))

  if corner_image.mode != 'RGB':
      corner_image = corner_image.convert('RGB')
  corner_image_bgr = np.array(corner_image)[:, :, ::-1]

  mask = corner_image.convert('L').point(lambda p: 255 if p > 128 else 0)

  mask_np = np.array(mask)

  frame_roi = frame[corner_y:corner_y+corner_height, corner_x:corner_x+corner_width]

  frame_roi_bgr = frame_roi[:, :, ::-1]
  result_bgr = np.where(mask_np[..., None] == 0, frame_roi_bgr, corner_image_bgr)

  frame[corner_y:corner_y+corner_height, corner_x:corner_x+corner_width] = result_bgr

  return frame

def detect_emotion(face_image):
    try:
        result = DeepFace.analyze(face_image, actions=['emotion'],enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion
    except Exception as e:
        print("Error:", str(e))
        return None

def main():
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()
    cap = cv2.VideoCapture(0)

    emotion = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        corner_image_paths = [
            "filters/angry.png",
            "filters/fear.png",
            "filters/angry.png",
            "filters/angry.png"
        ]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                face_image = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                emotion = detect_emotion(face_image)
                if emotion:
                    filter_path = f'filters/{emotion}.png'
                    overlay_image = cv2.imread(filter_path, -1)
                    cv2.putText(frame, emotion, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    overlay_resized = cv2.resize(overlay_image, (bbox[2], bbox[3]))
                    overlay_x, overlay_y = bbox[0], bbox[1]
                    overlay_alpha_s = overlay_resized[:, :, 3] / 255.0
                    overlay_alpha_l = 1.0 - overlay_alpha_s
                    for c in range(0, 3):
                        frame[overlay_y:overlay_y+bbox[3], overlay_x:overlay_x+bbox[2], c] = \
                            (overlay_alpha_s * overlay_resized[:, :, c] +
                            overlay_alpha_l * frame[overlay_y:overlay_y+bbox[3], overlay_x:overlay_x+bbox[2], c])                
        corner_positions = get_corner_positions(frame,corner_ratio)

        for i, corner_path in enumerate(corner_image_paths):
            frame = place_image_on_corner(frame.copy(), corner_path, corner_positions[i])

        # Display the processed frame
        # cv2.imshow('Video Feed', frame)

        # Return emotion and frame for each iteration
        print(frame)
        return emotion, frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

import cv2
from deepface import DeepFace

model = DeepFace.build_model("Emotion")

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# FrontEnd
font = cv2.FONT_HERSHEY_DUPLEX
font_scale = 4.0
font_thickness = 3
font_color = (255, 255, 255)

frame_size = 400

def resize_to_square(image, size):
    h, w = image.shape[:2]
    if h > w:
        top = 0
        bottom = 0
        left = (h - w) // 2
        right = (h - w) // 2
    else:
        top = (w - h) // 2
        bottom = (w - h) // 2
        left = 0
        right = 0
    square_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    square_image = cv2.resize(square_image, (size, size))
    return square_image

def get_emotion_detection_result():
    while True:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y + h, x:x + w]
            resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
            normalized_face = resized_face / 255.0
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)

            preds = model.predict(reshaped_face)[0]
            emotion_idx = preds.argmax()
            emotion = emotion_labels[emotion_idx]

            overlay = frame.copy()
            opacity = 0.9  # Reduced opacity for the text box

            # Create a transparent box for text
            cv2.rectangle(overlay, (0, frame_size -- 175), (600, frame_size -- 600), (0, 0, 0), -1)

            cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

            cv2.putText(frame, emotion, (10, frame_size - -300), font, font_scale, font_color, font_thickness)


# Apply emotion-based filter
            filter_path = f'filters/{emotion}.png'
            filter_img = cv2.imread(filter_path, -1)

            if filter_img is not None:
                filter_img = resize_to_square(filter_img, w)
                mask = filter_img[:, :, 3] / 255.0

                for c in range(3):
                    frame[y:y + h, x:x + w, c] = frame[y:y + h, x:x + w, c] * (1 - mask) + filter_img[:, :, c] * mask

        frame = resize_to_square(frame, frame_size)

        return frame

        # 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

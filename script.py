import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam


# loading pretrained model
model = load_model('model/fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# using FER2013 dataset, can only use these emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']

# load emoji images
emoji_images = {}
for emotion in emotion_labels:
    emoji_img = cv2.imread(f'emojis/{emotion}.png', cv2.IMREAD_UNCHANGED)  
    emoji_images[emotion] = emoji_img

# load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray_frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64, 64)) 
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)  # channel dimension
        roi = np.expand_dims(roi, axis=0)   # batch dimension


        # Ppredict most likely emotion
        prediction = model.predict(roi, verbose=0)[0]
        max_index = np.argmax(prediction)
        label = emotion_labels[max_index]

        # draw rectangle around face and put emoji over it
        emoji = emoji_images[label]

        if emoji is not None:
            # resize emoji to match face width and height
            emoji_resized = cv2.resize(emoji, (w, h), interpolation=cv2.INTER_AREA)

            if emoji_resized.shape[2] == 4:
                emoji_rgb = emoji_resized[:, :, :3]
                alpha_mask = emoji_resized[:, :, 3] / 255.0

                # our region of interest
                y1, y2 = y, y + h
                x1, x2 = x, x + w

                # coordinates are within frame bounds
                if y2 <= frame.shape[0] and x2 <= frame.shape[1]:
                    roi = frame[y1:y2, x1:x2]

                    # blend emoji with frame using alpha mask
                    for c in range(3): 
                        roi[:, :, c] = (alpha_mask * emoji_rgb[:, :, c] +
                                        (1 - alpha_mask) * roi[:, :, c])

                    frame[y1:y2, x1:x2] = roi
        else:
            # fallback in case emoji failed to load
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


    cv2.imshow("Facial Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

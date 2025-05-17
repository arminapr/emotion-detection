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
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

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

        # draw rectangle around face and put label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow("Facial Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import numpy as np
import cv2
import os

################################# KNN ##############################


def distance(x1, x2):
    return np.sqrt(sum(x1-x2)**2)


def knn(train, test, k=10):
    dist = []

    for i in range(train.shape[0]):
       
        ix = train[i, :-1]
        iy = train[i, -1]
      
        d = distance(test, ix)
        dist.append([d, iy])

    dist = sorted(dist, key=lambda x: x[0])[:k]
    
    labels = np.array(dist)[:, -1]
    
    output = np.unique(labels, return_counts=True)
    
    idx = np.argmax(output[1])
    return output[0][idx]

########################################################


# initialise the camera:
cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './our_data/'

face_data = []

label = []

class_id = 0

names = {}

# Data Preperation:
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path+fx)
        print('loaded '+fx)
        face_data.append(data_item)
        # Create Labels for the class:
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        label.append(target)

face_dataset = np.concatenate(face_data, axis=0)
label_dataset = np.concatenate(label, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print(label_dataset.shape)

train_dataset = np.concatenate((face_dataset, label_dataset), axis=1)
print(train_dataset.shape)

# Testing:
while True:
    ret, frame = cam.read()

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    if len(faces) == 0:
        continue

    for f in faces:
        x, y, w, h = f      
        offset = 10
       
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        output = knn(train_dataset, face_section.flatten())
        predicted_name = names[int(output)]
       
        cv2.putText(frame, predicted_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    
    cv2.imshow("Faces", frame)

    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

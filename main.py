import tkinter as tk
import settings
import cv2
from PIL import Image, ImageTk
import numpy as np
import imutils


def start_video():
    settings.start_video = True
    show_frame()


def stop_video():
    settings.start_video = False
    settings.start_processing = False
    lmain.config(image='')


def start_process():
    settings.start_processing = True


def stop_process():
    settings.start_processing = False


def show_frame():
    if not settings.start_video:
        return None

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=400)

    if settings.start_processing:
        frame = process_frame(frame)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


def process_frame(img):
    # grab the frame dimensions and convert it to a blob
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.2:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                                         confidence * 100)
            cv2.rectangle(img, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(img, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    return img


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("Loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
cap = cv2.VideoCapture(0)

window = tk.Tk()
window.title("DEMO")
window.geometry('700x420')
lbl = tk.Label(window, text="Pi Object Detection Module Demo", font=("Arial Bold", 24))
lbl.grid(column=1, row=0)
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=1, column=1, padx=10, pady=2)
lmain = tk.Label(imageFrame, text="Press Start Video")
lmain.grid(row=1, column=1)
startVideoStreamBtn = tk.Button(window, text="Start Video", command=start_video)
startVideoStreamBtn.grid(column=0, row=2, padx=15)
stopVideoStreamBtn = tk.Button(window, text="Stop Video", command=stop_video)
stopVideoStreamBtn.grid(column=0, row=3, padx=15)
startProcessBtn = tk.Button(window, text="Start Detection", command=start_process)
startProcessBtn.grid(column=1, row=2)
stopProcessBtn = tk.Button(window, text="Stop Detection", command=stop_process)
stopProcessBtn.grid(column=1, row=3)
window.mainloop()

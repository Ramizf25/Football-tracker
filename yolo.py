import cv2
import os

def initialise_yolo():
    '''
    Function to initialise the weights, architecture, and labels for the yolo model

    :return: yolo model, yolo model output layer names, model labels (ball, person, etc.)
    '''
    yolo_dir = os.path.abspath("./yolo")

    # load the labels, weights, and config for the yolo model
    labels_path = os.path.sep.join([yolo_dir, "coco.names"])
    weights_path = os.path.sep.join([yolo_dir, "yolov3.weights"])
    config_path = os.path.sep.join([yolo_dir, "yolov3.cfg"])

    # load the labels (as list), and model
    labels = open(labels_path).read().strip().split("\n")
    yolo_model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # get the output layers
    layer_names = yolo_model.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in yolo_model.getUnconnectedOutLayers()]

    return yolo_model, layer_names, labels

import cv2
import imutils

vs = cv2.VideoCapture('game.mp4')

while True:
    # grab the next frame in the video stream
    grabbed, frame = vs.read()

    # check to see if we have reached the end of the stream
    if frame is not None:
           
    # resize the frame (so we can process it faster)
        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

# use the yolo model to detect objects in image and get their bounding boxes
blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
yolo_model.setInput(blob)
layerOutputs = yolo_model.forward(ln)
boxes = []
confidences = []
classIDs = []

for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
        # get the class id and confidence for the object
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        if confidence > confidence_threshold:

            # get the bounding box for the object
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            # update our list of bounding box coordinates, confidences, and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# apply non-maxima suppression to refine the balls bounding box
idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, suppression_threshold)

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # we only care about the ball
        if labels[classIDs[i]] == "sports ball":

            # let the system know we found the ball
            ball_found_initially = True
            ball_found_in_frame = True
            object_lost_count = 0
            prediction_count = 0

            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # calculate the centre of the bounding box to draw the trace on the frame
            trace_location = (int(x + (w / 2)), int(y + (h / 2)))
from sklearn.linear_model import LinearRegression

x_train = [pt[0] for pt in tracked_points]
x_train.reverse()

# fit a simple linear regression model to predict the next x position
x_train = x_train[-20:]
times = [i for i in range(len(x_train))]

reg = LinearRegression().fit(np.array(times).reshape(-1, 1), np.array(x_train).reshape(-1, 1))
x = reg.predict(np.array([20]).reshape(1, -1))[0]

def quadratic_eqn(x, a, b, c):
    return (a * (x * x)) + (b * x) + c

y_train = [pt[1] for pt in tracked_points]
y_train.reverse()

# obtain values for a, b, c 
popt, pcov = curve_fit(quadratic_eqn, np.array(x_train[-20:]), np.array(y_train[-20:]))

# use our curve fit to predict the next y
y = quadratic_eqn(x, popt[0], popt[1], popt[2])

trace_location = (int(x + (w / 2)), int(y + (h / 2)))

for i in range(1, len(tracked_points)):
        # if either of the tracked points are None, ignore them
        if tracked_points[i - 1] is None or tracked_points[i] is None:
            continue
        # otherwise, draw a line connecting them
        cv2.line(frame, tracked_points[i - 1], tracked_points[i], (255, 0, 0), 2)

# write the frame to our output file
if writer is None:
    # initialize our video writer
    writer = cv2.VideoWriter(output_location,
                             cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                             30, (frame.shape[1],
                                  frame.shape[0]),
                             True)

# write the output frame to disk
writer.write(frame)

writer.release()
vs.release()

import cv2
import numpy as np
import pandas as pd
from skimage import measure
import tensorflow as tf
from bounding_box_sorting import bboxes_sort

model = tf.keras.models.load_model('C:/Users/srava/Downloads/2017-IWT4S-HDR_LP-dataset/my_model.h5')
predefined_class_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A',
                         11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: ' J', 20: ' K',
                         21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U',
                         31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}


def detect_rect(input_path):
    input_df = pd.read_csv(input_path)
    k = 1
    number_plates = []
    for i in range(input_df.shape[0]):
        text = []
        temp_boxes = []
        image = cv2.imread(input_df['image_path'][i])
        # bboxes = text_boxes_extraction(image)\
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        labels = measure.label(closing, neighbors=8, background=0)
        for label in np.unique(labels):
            if label == 0:
                continue
            labelMask = np.zeros(closing.shape, dtype='uint8')
            labelMask[labels == label] = 255
            cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0]
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
                aspectRatio = boxW / float(boxH)
                heightRatio = boxH / float(image.shape[0])
                keepAspectRatio = 0.1 < aspectRatio < 1.0
                keepHeight = 0.3 < heightRatio < 0.95
                if keepAspectRatio and keepHeight:
                    box = [boxX, boxY, boxX + boxW, boxY + boxH]
                    temp_boxes.append(box)
        bboxes = bboxes_sort(temp_boxes)
        for box in bboxes:
            try:
                temp_img = closing[box[1] - 5:box[3] + 5, box[0] - 5: box[2] + 5]
                temp_img = cv2.resize(temp_img, (28, 28))
                temp_img = temp_img.reshape((1, 28, 28, 1))
                temp_img = temp_img.astype('float32')
                temp_img = temp_img / 255.0
                predicted = model.predict(temp_img)
                predicted_class = predefined_class_dict[np.argmax(np.round(predicted), axis=1)[0]]
                text.append(predicted_class)
            except:
                pass
        number_plates.append("".join(text))
    input_df['Number'] = number_plates
    input_df.to_csv('extracted_numbers.csv', index=False)


def main():
    detect_rect('./trainVal.csv')


if __name__ == '__main__':
    main()

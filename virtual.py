import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
import cvzone
import pyautogui

cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
factor = 2
distance_tip = 50

detector = HandDetector(detectionCon=0.8)
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", 'BACKSPACE'],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "ENTER"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", '[', ']'],
        [" ", " ", " ", " ", "SPACE", "", "", "", "", ""]
        ]
finalText = ""


def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
                          20, rt=0)
        cv2.rectangle(img, button.pos, (x + w, y + h), (237, 182, 17), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img


class Button:
    def __init__(self, pos, text, size=(85, 85)):
        self.pos = pos
        self.size = size
        self.text = text


buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        if key == 'SPACE':
            buttonList.append(Button([100 * j + 50, 100 * i + 50], key, size=(300, 85)))
        elif key == 'ENTER' or key == 'BACKSPACE':
            buttonList.append(Button([100 * j + 50, 100 * i + 50], key, size=(220, 85)))
        elif key != " " and key != '':
            buttonList.append(Button([100 * j + 50, 100 * i + 50], key))


class Virtual:
    def __init__(self):
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(0)

    def run(self, app):
        global finalText

        success, img = self.cap.read()
        img = cv2.resize(img, (0, 0), fx=factor, fy=factor)
        img = cv2.flip(img, 1)

        img = detector.findHands(img)
        lmList, bboxInfo = detector.findPosition(img)
        img = drawAll(img, buttonList)

        if lmList:
            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (17, 171, 237), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    l, _, _ = detector.findDistance(8, 12, img, draw=False)

                    ## when clicked
                    if l < distance_tip:
                        pyautogui.press(button.text.lower())
                        cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 20, y + 65),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                        finalText += button.text
                        if len(finalText) > 28:
                            finalText = ''

                        sleep(0.15)

        cv2.rectangle(img, (50, 500), (1270, 600), (237, 182, 17), cv2.FILLED)
        cv2.putText(img, finalText.lower(), (60, 580),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

        cv2.imshow("Vitual Keyboard", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            if app is not None:
                app.quit()
            else:
                cv2.destroyAllWindows()


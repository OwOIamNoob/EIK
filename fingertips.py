import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_hands = mp.solutions.hands


class HandProcessing:
    def __init__(self):
        self.prev_left = {
            'thumb': (0, 0, 0),
            'index': (0, 0, 0),
            'middle': (0, 0, 0),
            'ring': (0, 0, 0),
            'pinky': (0, 0, 0)
        }

        self.prev_right = {
            'thumb': (0, 0, 0),
            'index': (0, 0, 0),
            'middle': (0, 0, 0),
            'ring': (0, 0, 0),
            'pinky': (0, 0, 0)
        }

        self.hands_coordinates = []
        self.hands = mp_hands.Hands()

    def get_hand_info(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.hands_coordinates = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                coor = {
                    'thumb': (
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z
                    ),
                    'index': (
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z
                    ),
                    'middle': (
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z
                    ),
                    'ring': (
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z
                    ),
                    'pinky': (
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z
                    )
                }
                self.hands_coordinates.append(coor)

        return img

    def press_detect(self, img):
        img = self.get_hand_info(img)
        action = []

        if len(self.hands_coordinates) == 2:
            left = self.hands_coordinates[0]
            right = self.hands_coordinates[1]

            if left['thumb'][0] > right['thumb'][0]:
                left, right = right, left

            left_thumb = (int(left['thumb'][0] * img.shape[1]), int(left['thumb'][1] * img.shape[0]))
            right_thumb = (int(right['thumb'][0] * img.shape[1]), int(right['thumb'][1] * img.shape[0]))

            img = cv2.putText(img, text='Left hand', fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                              org=left_thumb,
                              color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            img = cv2.putText(img, text='Right hand', fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                              org=right_thumb,
                              color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            line_y = img.shape[0] - 100

            img = cv2.line(img, pt1=(0, line_y), pt2=(img.shape[1], line_y), color=(0, 0, 255), thickness=2)

            def check_status(hand, prev_hand, type_hand, input_img):
                # nonlocal img
                for finger in hand.keys():
                    cur_y = hand[finger][1] * input_img.shape[0]
                    prev_y = prev_hand[finger][1] * input_img.shape[0]

                    if (cur_y - line_y) * (prev_y - line_y) < 0:
                        if cur_y < line_y:
                            action.append({
                                'finger': finger,
                                'type_hand': type_hand,
                                }
                            )

                            input_img = cv2.putText(input_img, text=finger + ' ' + type_hand + ': PRESS',
                                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                    fontScale=1,
                                                    org=(img.shape[0] // 2 - 100, img.shape[1] // 2),
                                                    color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

                return input_img

            img = check_status(left, self.prev_left, 'left', img)
            img = check_status(right, self.prev_right, 'right', img)

            self.prev_left = left
            self.prev_right = right

        return img, action


def run():
    laptop_cap = cv2.VideoCapture(0)
    laptop_hand = HandProcessing()

    while True:
        data, image = laptop_cap.read()

        image = cv2.flip(image, 1)

        image, action = laptop_hand.press_detect(image)

        cv2.imshow('Handtracker', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    laptop_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()

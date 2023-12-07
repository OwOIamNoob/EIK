import cv2
import fingertips as ft
import action as act
import image


class SuperVirtual:
    def __init__(self):
        self.laptop_cap = None
        self.mobile_cap = None
        self.laptop_hand = None
        self.mobile_hand = None

    def open(self):
        self.laptop_cap = cv2.VideoCapture(0)
        self.mobile_cap = cv2.VideoCapture(0)
        self.mobile_cap.open('http://192.168.43.1:8080/video')
        self.laptop_hand = ft.HandProcessing()
        self.mobile_hand = ft.HandProcessing()

    def run(self, app=None):

        _, laptop_img = self.laptop_cap.read()
        _, mobile_img = self.mobile_cap.read()
        laptop_img = cv2.flip(laptop_img, 1)

        mobile_img = cv2.flip(mobile_img, 0)
        mobile_img = cv2.flip(mobile_img, 1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if app is not None:
                app.quit()
            else:
                cv2.destroyAllWindows()

        if key == ord('p'):
            cv2.imwrite('key.png', mobile_img)
            process_img, act.key_info = image.image_process('key.png')
            cv2.imshow('Process Image', process_img)

        laptop_img, actions = self.laptop_hand.press_detect(laptop_img)
        mobile_img = self.mobile_hand.get_hand_info(mobile_img)

        mobile_img = act.draw_info(mobile_img)
        mobile_img = act.action_processing(actions, self.mobile_hand.hands_coordinates, mobile_img)

        cv2.imshow('Press Detection', laptop_img)
        cv2.imshow('Keyboard Extraction', mobile_img)


if __name__ == '__main__':
    SuperVirtual().run()

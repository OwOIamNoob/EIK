import cv2
import fingertips as ft
import action as act
import image

def run():
    laptop_cap = cv2.VideoCapture(0)
    mobile_cap = cv2.VideoCapture(0)
    mobile_cap.open('http://192.168.43.1:8080/video')
    laptop_hand = ft.HandProcessing()
    mobile_hand = ft.HandProcessing()

    key_info = None

    while True:
        _, laptop_img = laptop_cap.read()
        _, mobile_img = mobile_cap.read()
        laptop_img = cv2.flip(laptop_img, 1)

        mobile_img = cv2.flip(mobile_img, 0)
        mobile_img = cv2.flip(mobile_img, 1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.imwrite('key.png', mobile_img)
            process_img, act.key_info = image.image_process('key.png')
            cv2.imshow('Process Image', process_img)

        laptop_img, actions = laptop_hand.press_detect(laptop_img)
        mobile_img = mobile_hand.get_hand_info(mobile_img)

        mobile_img = act.draw_info(mobile_img)
        mobile_img = act.action_processing(actions, mobile_hand.hands_coordinates, mobile_img)

        cv2.imshow('Press Detection', laptop_img)
        cv2.imshow('Keyboard Extraction', mobile_img)



if __name__ == '__main__':
    run()

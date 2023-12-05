import cv2
import pyautogui

keyboard_keys = [
    '\t', '\n', '\r', ' ', '!', '"', '#', '$', '%', '&', "'", '(',
    ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
    '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~',
    'accept', 'add', 'alt', 'altleft', 'altright', 'apps', 'backspace',
    'browserback', 'browserfavorites', 'browserforward', 'browserhome',
    'browserrefresh', 'browsersearch', 'browserstop', 'capslock', 'clear',
    'convert', 'ctrl', 'ctrlleft', 'ctrlright', 'decimal', 'del', 'delete',
    'divide', 'down', 'end', 'enter', 'esc', 'escape', 'execute', 'f1', 'f10',
    'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f2', 'f20',
    'f21', 'f22', 'f23', 'f24', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
    'final', 'fn', 'hanguel', 'hangul', 'hanja', 'help', 'home', 'insert', 'junja',
    'kana', 'kanji', 'launchapp1', 'launchapp2', 'launchmail',
    'launchmediaselect', 'left', 'modechange', 'multiply', 'nexttrack',
    'nonconvert', 'num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6',
    'num7', 'num8', 'num9', 'numlock', 'pagedown', 'pageup', 'pause', 'pgdn',
    'pgup', 'playpause', 'prevtrack', 'print', 'printscreen', 'prntscrn',
    'prtsc', 'prtscr', 'return', 'right', 'scrolllock', 'select', 'separator',
    'shift', 'shiftleft', 'shiftright', 'sleep', 'space', 'stop', 'subtract', 'tab',
    'up', 'volumedown', 'volumemute', 'volumeup', 'win', 'winleft', 'winright', 'yen',
    'command', 'option', 'optionleft', 'optionright'
]

hand_type = {
    'left': 0,
    'right': 1,
}

key_info = {}


def draw_info(img):
    for text in key_info.keys():
        x, y, w, h = key_info[text]

        start_point = (x, y)
        end_point = (x + w, y + h)
        color = (255, 0, 0)
        thickness = 2

        img = cv2.rectangle(img, start_point, end_point, color, thickness)
        offset = 30
        img = cv2.putText(img, text=text,
                          fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                          fontScale=0.7,
                          org=(x + offset, y + offset),
                          color=(255, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    return img


def find_text(coor, image):
    for text in key_info.keys():
        x, y, w, h = key_info[text]

        if x < coor[0] * image.shape[1] < x + w and y < coor[1] * image.shape[0] < y + h:
            return text


def action_processing(actions, coors, img):
    if len(coors) == 2:
        for act in actions:
            finger = act['finger']
            type_hand = 0 if act['type_hand'] == 'left' else 1

            if coors[0][finger] > coors[1][finger]:
                type_hand = 1 - type_hand

            text = find_text(coors[type_hand][finger], img)

            img = cv2.putText(img,
                              text=f'Left: {finger} ->{text}' if type_hand == 0 else f'Right: {finger} -> {text}',
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=1.5,
                              org=(300, img.shape[0] - 100),
                              color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

            if text in keyboard_keys:
                pyautogui.press(text)

    return img

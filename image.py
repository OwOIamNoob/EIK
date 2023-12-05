import cv2
import numpy as np
import pytesseract


def image_process(path, thresh_area=(4000, 200000)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh1 = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray, 50, 150)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_img = img.copy()

    key_imgs = []
    cnt = 0

    # custom_config = r'--oem 3 --psm 6'
    custom_config = r'-l eng --psm 6'
    ans = {}

    for contour in contours:
        if cv2.contourArea(contour) < thresh_area[0] or cv2.contourArea(contour) > thresh_area[1]:
            continue
        cnt += 1

        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Nếu đường viền xấp xỉ có 4 góc (gần vuông)
        offset = 10
        x, y, w, h = cv2.boundingRect(approx)
        roi = thresh1[y + offset:y + h - offset, x + offset:x + w - offset]

        start_point = (x + offset, y + offset)  # Điểm bắt đầu của hình vuông
        end_point = (x + w - offset, y + h - offset)  # Điểm kết thúc của hình vuông
        color = (255, 0, 0)  # Màu đỏ trong định dạng BGR
        thickness = 2  # Độ dày của đường vẽ

        key_imgs.append(roi)

        output_img = cv2.rectangle(output_img, start_point, end_point, color, thickness)
        text = pytesseract.image_to_string(roi, config=custom_config)
        text = text.replace('\n', '')

        output_img = cv2.putText(output_img, text=text,
                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                 fontScale=0.7,
                                 org=(x + offset, y + offset),
                                 color=(255, 0, 255), thickness=1, lineType=cv2.LINE_AA)

        text = ''.join(text.lower().split(' '))
        if len(text) > 1 and text[0] == text[1]:
            text = ''  + text[0]

        ans[text] = (x, y, w, h)

    print(ans)

    return output_img, ans


if __name__ == '__main__':
    image_process('data/2.png')

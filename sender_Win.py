import pyautogui
import pyperclip
from pynput import mouse
from PIL import Image
import easyocr
import sys
import cv2
import numpy as np
import time
import ssl
from openai import OpenAI

ssl._create_default_https_context = ssl._create_unverified_context
client = OpenAI(api_key="在这里填写你的deepseek api-key", base_url="https://api.deepseek.com")
class RegionSelector:
    def __init__(self):
        self.clicks = []
    def on_click(self, x, y, button, pressed):
        if pressed:
            x = int(x)
            y = int(y)
            self.clicks.append((x, y))
            print(f"已捕获坐标：({x}, {y})")
            if len(self.clicks) >= 2:
                return False


def preprocess_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced)
def compare_images(img1, img2, threshold=50):
    if img1 is None or img2 is None:
        return True
    img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(img1_cv, img2_cv)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    change_pixels = np.sum(thresh) // 255
    return change_pixels > threshold


def GotResponse(msg):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system",
             "content": "用尽可能简短（只有几个字或一句话）来回复用户，如果看不懂则调侃用户。"},
            {"role": "user", "content": msg},
        ],
        stream=False
    )
    response_msg = response.choices[0].message.content
    print(f"AI回答：{response_msg}")
    pyperclip.copy(response_msg)
    pyautogui.hotkey('ctrl', 'v')
    pyautogui.press('enter')
def main():
    print("请点击屏幕上两个对角点选择区域（左上/右下）...")

    selector = RegionSelector()
    with mouse.Listener(on_click=selector.on_click) as listener:
        listener.join()

    if len(selector.clicks) < 2:
        print("错误：需要点击两个点！")

        sys.exit(1)
    x1, y1 = selector.clicks[0]
    x2, y2 = selector.clicks[1]
    left = min(x1, x2)
    top = min(y1, y2)
    width = abs(x1 - x2)
    height = abs(y1 - y2)

    reader = easyocr.Reader(['ch_sim', 'en'])
    base_screenshot = None
    previous_texts = set()

    try:
        while True:
            try:
                current_screenshot = pyautogui.screenshot(region=(left, top, width, height))

                if base_screenshot is None:
                    processed_img = preprocess_image(current_screenshot)
                    results = reader.readtext(np.array(processed_img))
                    current_texts = {result[1] for result in results}

                    if current_texts:
                        print("\n发现初始消息：")
                        msg = "".join(current_texts)
                        print(msg)
                        GotResponse(msg)
                        print("等待界面稳定...")
                        time.sleep(1.5)
                        base_screenshot = pyautogui.screenshot(region=(left, top, width, height))
                        previous_texts = current_texts.copy()
                else:

                    if compare_images(current_screenshot, base_screenshot):
                        print("\n检测到界面变化，分析新消息...")
                        processed_img = preprocess_image(current_screenshot)
                        results = reader.readtext(np.array(processed_img))
                        current_texts = {result[1] for result in results}
                        new_texts = current_texts - previous_texts

                        if new_texts:
                            print("发现新内容：")
                            msg = "".join(new_texts)
                            print(msg)
                            GotResponse(msg)
                            print("等待界面稳定...")
                            time.sleep(1.5)
                            base_screenshot = pyautogui.screenshot(region=(left, top, width, height))
                            previous_texts = current_texts.copy()
                        else:
                            print("界面变化但未识别到新文本")
                            base_screenshot = current_screenshot
                    else:
                        print("\n当前界面无变化")
                time.sleep(2)

            except Exception as e:
                print(f"处理过程中发生错误: {e}")
                time.sleep(2)

    except KeyboardInterrupt:
        print("\n程序已手动终止")


if __name__ == "__main__":
    main()

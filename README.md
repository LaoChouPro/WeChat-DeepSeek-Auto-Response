# WeChat-DeepSeek-Auto-Response
这是一个基于视觉的、非侵入式的微信自动回复程序，使用0封号风险，调用deepseek（或其它）API，基于OCR识别进行自动回复。

该项目是一个基于Python的微信自动回复工具，利用了图像识别、OCR（光学字符识别）、鼠标监听和AI接口来实现自动回复功能。
程序通过捕获指定区域的屏幕截图，识别其中的文本内容，并通过调用AI接口生成合适的回复。
最后，程序会自动将回复复制到剪贴板并通过模拟按键发送。

这样的逻辑同样也可用于其它有相同发送消息逻辑的网站或软件。

## 特性

- **区域选择**：通过点击屏幕选择截图区域（左上和右下角）。
- **图像预处理**：增强对比度，提升OCR识别的准确性。
- **图像比较**：比较截图变化，避免无意义的重复处理。
- **AI自动回复**：利用DeepSeek API根据识别到的消息生成自动回复。
- **自动粘贴与发送**：复制AI的回复到剪贴板并模拟按键操作发送。

## 安装与使用

### 依赖安装

首先，确保你已经安装了以下依赖：

- `pyautogui`：自动化控制鼠标键盘。
- `pyperclip`：用于复制文本到剪贴板。
- `pynput`：监听鼠标事件。
- `Pillow`：图像处理库。
- `easyocr`：OCR识别工具。
- `opencv-python`：用于图像处理和比较。
- `openai`：DeepSeek AI接口调用。

你可以通过以下命令安装所有依赖：

```bash
pip install pyautogui pyperclip pynput pillow easyocr opencv-python openai
```

### 获取API密钥

本工具通过DeepSeek AI接口提供自动回复功能。你需要在[DeepSeek](https://https://deepseek.com)注册并获取API密钥。将密钥替换在代码中的`client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com")`部分。

### 使用步骤

1. 运行脚本后，点击屏幕上的两个对角点来选择需要监视的区域（左上角和右下角）。
2. 程序会自动开始捕获该区域的屏幕截图，并进行OCR识别。
3. 当检测到新的文本变化时，程序会通过AI生成自动回复。
4. 自动将回复复制到剪贴板并发送到微信界面。

### 运行程序
您可以运行Auto_choose_new_sender.py ,它可以自动回复刚刚发送消息的人。如果此时有其它人发来了消息，它也会自动切换新发的人进行回复。
使用该程序时，应该点击四下屏幕，分别为：
1. 点击需要识别对方发送消息的内容位置的矩形的左上角；
2. 点击需要识别对方发送消息的内容位置的矩形的右下角；
3. 点击左侧用户栏中，最新会话的用户所处的位置（也就是最顶上）
4. 点击聊天窗口的信息输入位置。

win 和 macOS上的区别仅在于应用粘贴时，时command键还是ctrl键。
Auto_choose_new_sender.py可以在win上直接运行。
直接运行程序，观察终端输出。

## 代码解释

### 1. 区域选择与鼠标监听

通过`pynput`库监听鼠标事件，捕获两次点击的坐标，用于选择要监视的屏幕区域。
这个区域最好是人新发消息的文字区域，可以保证最好地仅识别对方发来的文字内容。

### 2. 图像预处理

程序会对捕获的屏幕截图进行对比度增强，以便OCR更好地识别文本。使用OpenCV的`CLAHE`（对比度限制自适应直方图均衡化）算法来进行图像处理。

### 3. 图像比较

程序会对新截图与上一张截图进行比较，如果有变化，则认为屏幕内容发生了更新，开始执行OCR识别。

### 4. OCR识别

使用`easyocr`进行文本识别，识别出的文本会被发送给AI接口，AI根据识别的文本生成回复。

### 5. AI自动回复

调用DeepSeek AI接口，通过给定的用户消息生成简短的自动回复。然后将回复内容复制到剪贴板并模拟键盘操作将其发送出去。

## 示例
你可以用鼠标指针选中聊天窗口中，对方接下来发送消息的小区域的矩形的左上和右下角。
假设监控区域中显示的消息为“你好，今天的天气怎么样？”，程序将识别到这个文本并调用AI生成简短的回复，比如“今天天气不错，适合出行”。此时，回复会自动被复制到剪贴板并发送。
使用效果截图：
![CleanShot 2025-02-20 at 08 25 29@2x](https://github.com/user-attachments/assets/2ae5fe58-3edf-4e3f-81be-a5edce3b82e0)
![CleanShot 2025-02-20 at 08 27 13@2x](https://github.com/user-attachments/assets/ac57e1c9-183b-4565-837f-b2cc3e15c814)


## 注意事项

- 确保你已经授权AI接口的API密钥。
- 调整`compare_images`函数中的阈值以适应不同的屏幕和内容变化。
- 程序适用于任何可用的屏幕区域，只要该区域内有明确的文本显示。
- 本程序使用时没有任何封号风险，因为它是纯基于视觉的，而非侵入式回复。

## 贡献

欢迎提出问题和贡献代码，若你有任何建议或遇到问题，可以提交issue或者pull request。

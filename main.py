import cv2
import numpy as np
import onnxruntime
import threading
import time
import serial
import threading
import time
from imutils.video import VideoStream
from pyzbar import pyzbar
import datetime
import imutils
import Adafruit_DHT as DHT
import RPi.GPIO as GPIO

Sensor = 11      #DHT11管脚
humiture = 17    #DHT位置
temperature = 0  #当前温度
humidity = 0     #当前湿度
CLASSES = ['people', 'fall']    #识别标签
cap = cv2.VideoCapture(0)       #打开摄像头
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))    #设置摄像头参数
#线程管理变量
fall_running = False
qr_running = False
receive_running = True
check_running = True
total_running = True
is_running = False
#模型位置
onnx_path = '/home//pi//PycharmProjects//pc8591_main//best.onnx'
fps_counter = 0  # 帧率计数器
start_time = time.time()  # 开始时间
path = []  # 用于存储路径
found = None  #二维码存储



# 发送消息
def send_message(message):
    global ser
    if message == "qian":
        ser.send("DW")
        print("[INFO] instruction: 前进")
    if message == "hou":
        ser.send("DS")
        print("[INFO] instruction: 后退")
    if message == "zuo":
        ser.send("DA")
        print("[INFO] instruction: 左转")
    if message == "you":
        ser.send("DD")
        print("[INFO] instruction: 右转")
    if message == "stop":
        ser.send("DT")
        print("[INFO] instruction: 停止")


# 封装resize函数
def resize_img_keep_ratio(img, target_size):
    old_size = img.shape[0:2]  # 原始图像大小
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))  # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    new_size = tuple([int(i * ratio) for i in old_size])  # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img, (new_size[1], new_size[0]))  # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1]  # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0]  # 计算需要填充的像素数目（图像的高这一维度上）
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    return img_new


class SerialConnection:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.receive_message = None
        self.lock = threading.Lock()

    def open(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate)
            if self.serial.isOpen():
                print("[INFO] Serial port is open")
        except serial.SerialException as e:
            print("[INFO] Failed to open serial port: ", str(e))
            self.serial = None
        return self.serial is not None

    def close(self):
        if self.serial:
            self.serial.close()
        if not self.serial.isOpen():
            print("[INFO] Serial port is closed")

    def send(self, message):
        if self.serial:
            try:
                self.serial.write(message.encode('gb2312'))
                return True
            except serial.SerialException as e:
                print("[INFO] Failed to send message: ", str(e))
        return False

    def receive(self):
        global receive_running
        if self.serial:
            while receive_running:
                try:
                    self.lock.acquire()
                    size = self.serial.inWaiting()
                    if size != 0:
                        self.receive_message = self.serial.read(size)
                        print("[INFO]receiving: ", self.receive_message)
                        self.serial.flushInput()
                        time.sleep(0.1)
                except serial.SerialException as e:
                    print("Failed to receive message: ", str(e))
                finally:
                    self.lock.release()
            print("receive stop")
        return None


class YOLOV5():
    def __init__(self, onnxpath):
        self.onnx_session = onnxruntime.InferenceSession(onnxpath)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    # -------------------------------------------------------
    #   获取输入输出的名字
    # -------------------------------------------------------
    def get_input_name(self):
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self):
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    # -------------------------------------------------------
    #   输入图像
    # -------------------------------------------------------
    def get_input_feed(self, img_tensor):
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = img_tensor
        return input_feed

    # -------------------------------------------------------
    #   1.cv2读取图像并resize
    #	2.图像转BGR2RGB和HWC2CHW
    #	3.图像归一化
    #	4.图像增加维度
    #	5.onnx_session 推理
    # -------------------------------------------------------
    def inference(self, img):
        or_img = resize_img_keep_ratio(img, (640, 640))
        img = or_img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None, input_feed)[0]
        return pred, or_img


model = YOLOV5(onnx_path)      #onnx模型初始化


# dets:  array [x,6] 6个值分别为x1,y1,x2,y2,score,class
# thresh: 阈值
def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # -------------------------------------------------------
    #   计算框的面积
    #	置信度从大到小排序
    # -------------------------------------------------------
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)
        # -------------------------------------------------------
        #   计算相交面积
        #	1.相交
        #	2.不相交
        # -------------------------------------------------------
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        # -------------------------------------------------------
        #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
        #	IOU小于thresh的框保留下来
        # -------------------------------------------------------
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep


def xywh2xyxy(x):
    # [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def filter_box(org_box, conf_thres, iou_thres):  # 过滤掉无用的框
    # -------------------------------------------------------
    #   删除为1的维度
    #	删除置信度小于conf_thres的BOX
    # -------------------------------------------------------
    org_box = np.squeeze(org_box)
    print("[INFO] reliability: ", max(org_box[..., 4]))
    conf = org_box[..., 4] > conf_thres
    box = org_box[conf == True]
    # -------------------------------------------------------
    #	通过argmax获取置信度最大的类别
    # -------------------------------------------------------
    cls_cinf = box[..., 5:]
    cls = []
    for i in range(len(cls_cinf)):
        cls.append(int(np.argmax(cls_cinf[i])))
    all_cls = list(set(cls))
    # -------------------------------------------------------
    #   分别对每个类别进行过滤
    #	1.将第6列元素替换为类别下标
    #	2.xywh2xyxy 坐标转换
    #	3.经过非极大抑制后输出的BOX下标
    #	4.利用下标取出非极大抑制后的BOX
    # -------------------------------------------------------

    output = []
    for i in range(len(all_cls)):
        curr_cls = all_cls[i]
        curr_cls_box = []
        curr_out_box = []
        for j in range(len(cls)):
            if cls[j] == curr_cls:
                box[j][5] = curr_cls
                curr_cls_box.append(box[j][:6])
        curr_cls_box = np.array(curr_cls_box)
        # curr_cls_box_old = np.copy(curr_cls_box)
        curr_cls_box = xywh2xyxy(curr_cls_box)
        curr_out_box = nms(curr_cls_box, iou_thres)
        for k in curr_out_box:
            output.append(curr_cls_box[k])
    output = np.array(output)
    return output


def draw(image, box_data):
    # -------------------------------------------------------
    #	取整，方便画框
    # -------------------------------------------------------
    boxes = box_data[..., :4].astype(np.int32)
    scores = box_data[..., 4]
    classes = box_data[..., 5].astype(np.int32)

    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print('[INFO] class: {}, score: {}'.format(CLASSES[cl], score))
        print('[INFO] box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def check():
    #----------全局变量-----------
    global cap
    global model
    global fps_counter
    global start_time
    global check_running
    global found
    global qr_running
    global fall_running
    global temperature
    global humidity
    global ser
    #-----------------------------

    while check_running:                  #摄像头运行中
        #------------------------------发送温湿度信息---------------------------
        current_humidity, current_temperature = DHT.read_retry(Sensor, humiture)
        if current_humidity is not None and current_temperature is not None:
            pass
        else:
            print("[INFO] Failed to get reading. Try again!")

        if current_humidity != humidity or current_temperature != temperature:  #如果不同则更新并发送给单片机
            temperature = current_temperature
            humidity = current_humidity
            print("[INFO] Temperature = {0:0.1f}*C Humidity = {1:0.1f}%"
                  .format(temperature, humidity))
            ser.send("t" + str(temperature))
            ser.send("h" + str(humidity))
        #-----------------------------------------------------------------------

        #-------------------------------摔倒检测进程-----------------------------
        while fall_running:
            ret, frame = cap.read()
            if ret:
                fps_counter += 1  # 计算帧数
                output, or_img = model.inference(frame)  # 模型推理
                outbox = filter_box(output, 0.5, 0.5)  # 过滤框
                if (len(outbox) != 0):  # 检测到目标
                    draw(or_img, outbox)  # 画框
                else:  # 未检测到目标
                    pass
                if (time.time() - start_time) != 0:  # 实时显示帧数
                    cv2.putText(or_img, "FPS {0}".format(float('%.1f' % (fps_counter / (time.time() - start_time)))),
                                (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2)
                cv2.imshow('fall_detect.jpg', or_img)  # 显示图片
                print("[INFO] FPS: ", fps_counter / (time.time() - start_time))
                fps_counter = 0
                start_time = time.time()
                key = cv2.waitKey(1) & 0xFF
            else:
                print('[INFO] cap error')
                break
        #------------------------------------------------------------------------
        #-------------------------------二维码检测进程----------------------------
        while qr_running:
            ret, frame = cap.read()
            if ret:
                frame = imutils.resize(frame, width=400)
                barcodes = pyzbar.decode(frame)
                for barcode in barcodes:
                    (x, y, w, h) = barcode.rect
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    barcodeData = barcode.data.decode("utf-8")
                    barcodeType = barcode.type
                    text = "{} ({})".format(barcodeData, barcodeType)
                    if found == None or found != barcodeData:
                        found = barcodeData
                        print(text)
                        send_message(barcodeData)
                cv2.imshow("Barcode Scanner", frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                print('cap error')
                break
        #------------------------------------------------------------------------

def loop():
    #-------------------全局变量-------------------
    global qr_running
    global fall_running
    global check_running
    global total_running
    global is_running
    global ser
    #----------------------------------------------
    
    #线程管理模块
    while not qr_running and not fall_running and check_running:
        time.sleep(0.1)
        key = input("input states:")
        if key == "m":
            fall_running = True
            qr_running = False
            is_running = True
            print("[INFO] starting fall detect...")
        elif key == "q":
            fall_running = False
            qr_running = True
            is_running = True
            print("[INFO] starting QRcode detect...")
        elif key == "e":
            print("[INFO] exiting program...")
            check_running = False
            is_running = False
            destory()
            total_running = False

    #输入指令控制模块
    while is_running:
        key = input("input command:")
        if key == "e":
            fall_running = False
            qr_running = False
            is_running = False
        if len(key) >= 2 and key[0] == 's':
            ser.send(key[1:])


def destory():
    global qr_running
    global check_running
    global receive_running

    check_running = False
    qr_running = False
    receive_running = False
    check.join()  # 等待线程结束
    receive_thread.join()
    ser.close()
    cap.release()
    GPIO.cleanup()
    cv2.destroyAllWindows()


def path_get():
    global ser
    global path
    path_length = len(path)
    if ser.receive_message[0] == '1':  # 标志符号
        if ser.receive_message[1] != path[path_length - 1][0] or len(path) == 0:
            if path[path_length - 1][1] == 0:
                path.pop()
            tmp = [ser.receive_message[1], ser.receive_message[2]]
            path.append(tmp)
        else:
            path[path_length - 1][1] += ser.receive_message[0]


def path_back():
    global ser
    global path
    while len(path) >= 1:
        rotation = path[-1][0]


if __name__ == "__main__":
    try:
        global ser
        ser = SerialConnection('/dev/ttyAMA0', 115200)  # create serial
        ser.open()
        receive_thread = threading.Thread(target=ser.receive)  # create receive thread
        receive_thread.start()
        check = threading.Thread(target=check)
        check.start()
        while total_running:
            loop()
    except KeyboardInterrupt:
        destory()




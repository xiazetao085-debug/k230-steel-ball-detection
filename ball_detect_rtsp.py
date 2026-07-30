import gc
import os
import sys
import time
import uctypes
import _thread
import network
import socket
import multimedia as mm
import cv_lite

from libs.YOLO import YOLOv8
from media.sensor import *
from media.vencoder import *
from media.media import *


# K230D 热点参数
AP_SSID = "K230D_BALL"
AP_KEY = "12345678"

# RTSP 参数
RTSP_PORT = 8554
RTSP_SESSION = "test"
RTSP_WIDTH = 640
RTSP_HEIGHT = 480
RTSP_FPS = 20
RTSP_BIT_RATE = 1200

# 电脑端叠加播放器通过此 UDP 端口自动注册并接收识别结果。
UDP_PORT = 10000

# 钢球识别参数
DETECT_WIDTH = 640
DETECT_HEIGHT = 480
IMAGE_SHAPE = [DETECT_HEIGHT, DETECT_WIDTH]
DETECT_EVERY_N_FRAMES = 2
OUTPUT_EVERY_N_DETECTIONS = 1
DETECTOR_MODE = "yolo"

# YOLOv8 钢球模型参数
YOLO_KMODEL_PATH = "/sdcard/steel_ball_yolov8n_320.kmodel"
YOLO_LABELS = ["steel_ball"]
YOLO_MODEL_INPUT_SIZE = [320, 320]
YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_NMS_THRESHOLD = 0.45
YOLO_MAX_BOXES = 10
YOLO_MIN_BOX_SIZE = 12
YOLO_MAX_BOX_SIZE = 60
YOLO_MIN_ASPECT_RATIO = 0.65

# 水管在画面中的位置。
# 默认假设水管横向经过画面中央；若水管竖直，将 PIPE_AXIS 改为 "y"。
PIPE_AXIS = "x"
PIPE_ROI_X = 0
PIPE_ROI_Y = 125
PIPE_ROI_WIDTH = 500
PIPE_ROI_HEIGHT = 150
PIPE_CENTER_TOLERANCE = 65
POSITION_REVERSED = False

DP = 1
MIN_DIST = 20
PARAM1 = 100
PARAM2 = 24
MIN_RADIUS = 7
MAX_RADIUS = 24
EXPECTED_RADIUS = 11

# 连续检测和位置平滑参数。
CONFIRM_DETECTIONS = 2
MISS_HOLD_DETECTIONS = 3
MAX_TRACK_JUMP = 100
FILTER_ALPHA = 0.35


def start_wifi_ap():
    ap = network.WLAN(network.AP_IF)
    if not ap.active():
        ap.active(True)
    ap.config(ssid=AP_SSID, key=AP_KEY)
    time.sleep(1)

    ip_address = ap.ifconfig()[0]
    print("[WiFi] SSID:", AP_SSID)
    print("[WiFi] KEY:", AP_KEY)
    print("[RTSP] URL: rtsp://%s:%d/%s" %
          (ip_address, RTSP_PORT, RTSP_SESSION))
    return ap, ip_address


class BallUdpServer:
    def __init__(self, ip_address, port):
        self.client_address = None
        self.socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

        try:
            self.socket.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_REUSEADDR,
                1
            )
        except BaseException:
            pass

        address_info = socket.getaddrinfo(
            ip_address,
            port
        )
        self.socket.bind(address_info[0][-1])
        self.socket.settimeout(0)
        print("[UDP] waiting for viewer on port:", port)

    def poll_client(self):
        for _ in range(3):
            try:
                data, address = self.socket.recvfrom(64)
            except OSError:
                return

            if data.startswith(b"HELLO"):
                if self.client_address != address:
                    self.client_address = address
                    print(
                        "\n[UDP] viewer connected: %s:%d" %
                        (address[0], address[1])
                    )

                try:
                    self.socket.sendto(
                        b"K230D_BALL_READY",
                        address
                    )
                except OSError:
                    self.client_address = None

    def send(self, message):
        self.poll_client()
        if self.client_address is None:
            return

        try:
            self.socket.sendto(
                (message + "\n").encode(),
                self.client_address
            )
        except OSError as e:
            print("\n[UDP] send failed:")
            sys.print_exception(e)
            self.client_address = None

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        self.client_address = None


class SensorRtspServer:
    """将 Sensor 的独立 YUV 通道绑定到 VENC，并持续发送 RTSP。"""

    def __init__(self, sensor, sensor_chn,
                 width=640, height=480,
                 session_name="test", port=8554,
                 bit_rate=1200):
        self.sensor = sensor
        self.sensor_chn = sensor_chn
        self.width = ALIGN_UP(width, 16)
        self.height = height
        self.session_name = session_name
        self.port = port
        self.bit_rate = bit_rate

        self.encoder = Encoder()
        self.rtspserver = mm.rtsp_server()
        self.link = None

        self.encoder_created = False
        self.encoder_started = False
        self.server_started = False
        self.sensor_started = False
        self.start_stream = False
        self.runthread_over = True
        self.sent_frames = 0

    def configure(self):
        self.encoder.SetOutBufs(8, self.width, self.height)
        chn_attr = ChnAttrStr(
            self.encoder.PAYLOAD_TYPE_H264,
            self.encoder.H264_PROFILE_MAIN,
            self.width,
            self.height,
            bit_rate=self.bit_rate
        )
        self.encoder.Create(chn_attr)
        self.encoder_created = True

        sensor_src = self.sensor.bind_info(chn=self.sensor_chn)["src"]
        venc_dst = (
            VIDEO_ENCODE_MOD_ID,
            VENC_DEV_ID,
            self.encoder.chn
        )
        self.link = MediaManager.link(sensor_src, venc_dst)

    def start(self):
        if not self.encoder_created:
            raise RuntimeError("configure RTSP server before start")
        if self.start_stream:
            return

        self.rtspserver.rtspserver_init(self.port)
        self.rtspserver.rtspserver_createsession(
            self.session_name,
            mm.multi_media_type.media_h264,
            False
        )
        self.rtspserver.rtspserver_start()
        self.server_started = True

        self.encoder.Start()
        self.encoder_started = True

        self.sensor.run()
        self.sensor_started = True

        self.start_stream = True
        self.runthread_over = False
        _thread.start_new_thread(self._stream_thread, ())

    def _stream_thread(self):
        stream_data = StreamData()

        try:
            while self.start_stream:
                os.exitpoint()
                self.encoder.GetStream(stream_data)
                try:
                    for pack_idx in range(0, stream_data.pack_cnt):
                        pack = bytes(uctypes.bytearray_at(
                            stream_data.data[pack_idx],
                            stream_data.data_size[pack_idx]
                        ))
                        self.rtspserver.rtspserver_sendvideodata(
                            self.session_name,
                            pack,
                            stream_data.data_size[pack_idx],
                            1000
                        )
                finally:
                    self.encoder.ReleaseStream(stream_data)

                self.sent_frames += 1
        except BaseException as e:
            print("\n[RTSP] stream thread exception:")
            sys.print_exception(e)
        finally:
            self.start_stream = False
            self.runthread_over = True
            print("\n[RTSP] stream thread stopped")

    def stop(self):
        self.start_stream = False

        wait_count = 0
        while not self.runthread_over and wait_count < 50:
            time.sleep(0.1)
            wait_count += 1

        if self.sensor_started:
            self.sensor.stop()
            self.sensor_started = False

        if self.link is not None:
            del self.link
            self.link = None

        if self.encoder_started:
            self.encoder.Stop()
            self.encoder_started = False

        if self.encoder_created:
            self.encoder.Destroy()
            self.encoder_created = False

        if self.server_started:
            self.rtspserver.rtspserver_stop()
            self.rtspserver.rtspserver_deinit()
            self.server_started = False


def configure_sensor():
    # 使用和板载 RTSP 例程相同的自动传感器选择方式。
    sensor = Sensor()
    sensor.reset()

    # 通道 0：YOLO 使用 CHW 平面 RGBP888；霍夫圆使用打包 RGB888。
    sensor.set_framesize(
        width=DETECT_WIDTH,
        height=DETECT_HEIGHT,
        chn=CAM_CHN_ID_0
    )
    if DETECTOR_MODE == "yolo":
        sensor.set_pixformat(
            Sensor.RGBP888,
            chn=CAM_CHN_ID_0
        )
    else:
        sensor.set_pixformat(
            Sensor.RGB888,
            chn=CAM_CHN_ID_0
        )

    # 通道 1：YUV420SP，直接绑定硬件编码器。
    sensor.set_framesize(
        width=RTSP_WIDTH,
        height=RTSP_HEIGHT,
        alignment=12,
        chn=CAM_CHN_ID_1
    )
    sensor.set_pixformat(
        Sensor.YUV420SP,
        chn=CAM_CHN_ID_1
    )

    # 当前固件支持独立设置通道帧率；失败时仍可按传感器默认帧率运行。
    try:
        sensor._set_chn_fps(
            chn=CAM_CHN_ID_1,
            fps=RTSP_FPS
        )
    except BaseException as e:
        print("[RTSP] set channel FPS failed, use sensor default:")
        sys.print_exception(e)

    return sensor


def filter_pipe_candidates(circles):
    candidates = []
    roi_right = PIPE_ROI_X + PIPE_ROI_WIDTH
    roi_bottom = PIPE_ROI_Y + PIPE_ROI_HEIGHT

    if PIPE_AXIS == "x":
        centerline = PIPE_ROI_Y + PIPE_ROI_HEIGHT // 2
    elif PIPE_AXIS == "y":
        centerline = PIPE_ROI_X + PIPE_ROI_WIDTH // 2
    else:
        raise ValueError("PIPE_AXIS must be x or y")

    for index in range(0, len(circles), 3):
        x = int(circles[index])
        y = int(circles[index + 1])
        radius = int(circles[index + 2])

        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            continue
        if x - radius < PIPE_ROI_X or x + radius >= roi_right:
            continue
        if y - radius < PIPE_ROI_Y or y + radius >= roi_bottom:
            continue

        if PIPE_AXIS == "x":
            cross_error = abs(y - centerline)
        else:
            cross_error = abs(x - centerline)

        if cross_error > PIPE_CENTER_TOLERANCE:
            continue

        candidates.append((x, y, radius))

    return candidates


def filter_yolo_candidates(detections):
    candidates = []
    if not detections or len(detections) < 3:
        return candidates

    boxes = detections[0]
    class_ids = detections[1]
    scores = detections[2]
    roi_right = PIPE_ROI_X + PIPE_ROI_WIDTH
    roi_bottom = PIPE_ROI_Y + PIPE_ROI_HEIGHT

    if PIPE_AXIS == "x":
        centerline = PIPE_ROI_Y + PIPE_ROI_HEIGHT // 2
    elif PIPE_AXIS == "y":
        centerline = PIPE_ROI_X + PIPE_ROI_WIDTH // 2
    else:
        raise ValueError("PIPE_AXIS must be x or y")

    for index in range(len(boxes)):
        if int(class_ids[index]) != 0:
            continue
        if float(scores[index]) < YOLO_CONFIDENCE_THRESHOLD:
            continue

        box = boxes[index]
        x = float(box[0])
        y = float(box[1])
        width = float(box[2])
        height = float(box[3])
        if width < YOLO_MIN_BOX_SIZE or height < YOLO_MIN_BOX_SIZE:
            continue
        if width > YOLO_MAX_BOX_SIZE or height > YOLO_MAX_BOX_SIZE:
            continue

        long_side = max(width, height)
        short_side = min(width, height)
        if short_side / long_side < YOLO_MIN_ASPECT_RATIO:
            continue

        center_x = int(x + width * 0.5 + 0.5)
        center_y = int(y + height * 0.5 + 0.5)
        radius = int((width + height) * 0.25 + 0.5)

        if center_x < PIPE_ROI_X or center_x >= roi_right:
            continue
        if center_y < PIPE_ROI_Y or center_y >= roi_bottom:
            continue

        if PIPE_AXIS == "x":
            cross_error = abs(center_y - centerline)
        else:
            cross_error = abs(center_x - centerline)
        if cross_error > PIPE_CENTER_TOLERANCE:
            continue

        candidates.append((center_x, center_y, radius))

    return candidates


def circle_distance_sq(first, second):
    dx = first[0] - second[0]
    dy = first[1] - second[1]
    return dx * dx + dy * dy


class BallTracker:
    def __init__(self):
        self.valid = False
        self.status = "SEARCH"
        self.x = 0.0
        self.y = 0.0
        self.radius = 0.0
        self.pending = None
        self.pending_hits = 0
        self.misses = 0

    def _centerline_error(self, candidate):
        if PIPE_AXIS == "x":
            centerline = PIPE_ROI_Y + PIPE_ROI_HEIGHT // 2
            return abs(candidate[1] - centerline)

        centerline = PIPE_ROI_X + PIPE_ROI_WIDTH // 2
        return abs(candidate[0] - centerline)

    def _select_candidate(self, candidates):
        if not candidates:
            return None

        reference = None
        if self.valid:
            reference = (self.x, self.y, self.radius)
        elif self.pending is not None:
            reference = self.pending

        selected = None
        selected_score = None

        for candidate in candidates:
            score = self._centerline_error(candidate) * 2
            score += abs(candidate[2] - EXPECTED_RADIUS) * 4

            if reference is not None:
                score += circle_distance_sq(candidate, reference)

            if selected is None or score < selected_score:
                selected = candidate
                selected_score = score

        return selected

    def _start_pending(self, candidate):
        self.pending = candidate
        self.pending_hits = 1
        self.status = "CONFIRM"

    def update(self, candidates):
        candidate = self._select_candidate(candidates)
        max_jump_sq = MAX_TRACK_JUMP * MAX_TRACK_JUMP

        if self.valid:
            current = (self.x, self.y, self.radius)
            if (candidate is not None and
                    circle_distance_sq(candidate, current) <= max_jump_sq):
                keep = 1.0 - FILTER_ALPHA
                self.x = self.x * keep + candidate[0] * FILTER_ALPHA
                self.y = self.y * keep + candidate[1] * FILTER_ALPHA
                self.radius = (
                    self.radius * keep +
                    candidate[2] * FILTER_ALPHA
                )
                self.misses = 0
                self.status = "TRACK"
                return

            self.misses += 1
            if self.misses <= MISS_HOLD_DETECTIONS:
                self.status = "HOLD"
                return

            self.valid = False
            self.pending = None
            self.pending_hits = 0
            self.status = "LOST"

            if candidate is not None:
                self._start_pending(candidate)
            return

        if candidate is None:
            self.pending = None
            self.pending_hits = 0
            self.status = "SEARCH"
            return

        if (self.pending is None or
                circle_distance_sq(candidate, self.pending) > max_jump_sq):
            self._start_pending(candidate)
            return

        self.pending = candidate
        self.pending_hits += 1
        self.status = "CONFIRM"

        if self.pending_hits >= CONFIRM_DETECTIONS:
            self.x = float(candidate[0])
            self.y = float(candidate[1])
            self.radius = float(candidate[2])
            self.valid = True
            self.misses = 0
            self.status = "TRACK"

    def get_ball(self):
        if not self.valid:
            return None

        return (
            int(self.x + 0.5),
            int(self.y + 0.5),
            int(self.radius + 0.5)
        )


def get_axis_position(ball):
    if PIPE_AXIS == "x":
        axis_pixel = ball[0]
        roi_start = PIPE_ROI_X
        roi_length = PIPE_ROI_WIDTH
    else:
        axis_pixel = ball[1]
        roi_start = PIPE_ROI_Y
        roi_length = PIPE_ROI_HEIGHT

    ratio = (axis_pixel - roi_start) / float(roi_length)
    if ratio < 0.0:
        ratio = 0.0
    elif ratio > 1.0:
        ratio = 1.0

    centered_position = ratio * 2.0 - 1.0
    if POSITION_REVERSED:
        centered_position = -centered_position

    return axis_pixel, centered_position


def build_detection_message(tracker, candidate_count,
                            raw_candidate_count,
                            fps, sent_frames, sequence):
    ball = tracker.get_ball()
    if ball is None:
        axis_pixel = 0
        position = 0.0
        ball = (0, 0, 0)
        valid = 0
    else:
        axis_pixel, position = get_axis_position(ball)
        valid = 1

    return (
        "BALL,%d,%d,%d,%.3f,%d,%d,%d,%s,%d,%.1f,%d,"
        "%s,%d,%d,%d,%d,%d,%d,%s,%d" % (
            valid,
            sequence,
            axis_pixel,
            position,
            ball[0],
            ball[1],
            ball[2],
            tracker.status,
            candidate_count,
            fps,
            sent_frames,
            PIPE_AXIS,
            PIPE_ROI_X,
            PIPE_ROI_Y,
            PIPE_ROI_WIDTH,
            PIPE_ROI_HEIGHT,
            DETECT_WIDTH,
            DETECT_HEIGHT,
            DETECTOR_MODE,
            raw_candidate_count
        )
    )


def main():
    os.exitpoint(os.EXITPOINT_ENABLE)

    wifi_ap = None
    udp_server = None
    sensor = None
    rtsp_server = None
    yolo_detector = None

    try:
        wifi_ap, hotspot_ip = start_wifi_ap()
        udp_server = BallUdpServer(
            hotspot_ip,
            UDP_PORT
        )
        sensor = configure_sensor()

        rtsp_server = SensorRtspServer(
            sensor=sensor,
            sensor_chn=CAM_CHN_ID_1,
            width=RTSP_WIDTH,
            height=RTSP_HEIGHT,
            session_name=RTSP_SESSION,
            port=RTSP_PORT,
            bit_rate=RTSP_BIT_RATE
        )
        rtsp_server.configure()
        rtsp_server.start()

        if DETECTOR_MODE == "yolo":
            yolo_detector = YOLOv8(
                task_type="detect",
                mode="video",
                kmodel_path=YOLO_KMODEL_PATH,
                labels=YOLO_LABELS,
                rgb888p_size=[DETECT_WIDTH, DETECT_HEIGHT],
                model_input_size=YOLO_MODEL_INPUT_SIZE,
                display_size=[DETECT_WIDTH, DETECT_HEIGHT],
                conf_thresh=YOLO_CONFIDENCE_THRESHOLD,
                nms_thresh=YOLO_NMS_THRESHOLD,
                max_boxes_num=YOLO_MAX_BOXES,
                debug_mode=0
            )
            yolo_detector.config_preprocess()
            print("[BALL] detector: YOLOv8 KModel")
            print("[BALL] model:", YOLO_KMODEL_PATH)
        elif DETECTOR_MODE == "hough":
            print("[BALL] detector: Hough circle")
        else:
            raise ValueError("DETECTOR_MODE must be yolo or hough")

        print("[RTSP] open rtsp://%s:%d/%s" %
              (hotspot_ip, RTSP_PORT, RTSP_SESSION))
        print("[BALL] detection started")
        print("[BALL] pipe axis=%s roi=(%d,%d,%d,%d)" % (
            PIPE_AXIS,
            PIPE_ROI_X,
            PIPE_ROI_Y,
            PIPE_ROI_WIDTH,
            PIPE_ROI_HEIGHT
        ))
        print("[BALL] pos_norm: -1=start, 0=center, +1=end")
        print("[UDP] PC overlay port:", UDP_PORT)

        # 丢弃前几帧，让自动曝光和白平衡稳定。
        for _ in range(10):
            sensor.snapshot(chn=CAM_CHN_ID_0)

        clock = time.clock()
        frame_count = 0
        detection_count = 0
        tracker = BallTracker()

        while True:
            clock.tick()
            os.exitpoint()

            if not rtsp_server.start_stream:
                raise RuntimeError("RTSP stream thread stopped")

            udp_server.poll_client()
            img = sensor.snapshot(chn=CAM_CHN_ID_0)
            frame_count += 1

            if frame_count % DETECT_EVERY_N_FRAMES == 0:
                img_np = img.to_numpy_ref()
                if DETECTOR_MODE == "yolo":
                    detections = yolo_detector.run(img_np)
                    if detections and len(detections) >= 1:
                        raw_candidate_count = len(detections[0])
                    else:
                        raw_candidate_count = 0
                    candidates = filter_yolo_candidates(detections)
                else:
                    circles = cv_lite.rgb888_find_circles(
                        IMAGE_SHAPE,
                        img_np,
                        DP,
                        MIN_DIST,
                        PARAM1,
                        PARAM2,
                        MIN_RADIUS,
                        MAX_RADIUS
                    )
                    raw_candidate_count = len(circles) // 3
                    candidates = filter_pipe_candidates(circles)
                tracker.update(candidates)
                detection_count += 1

            if (frame_count % DETECT_EVERY_N_FRAMES == 0 and
                    detection_count % OUTPUT_EVERY_N_DETECTIONS == 0):
                message = build_detection_message(
                    tracker,
                    len(candidates),
                    raw_candidate_count,
                    clock.fps(),
                    rtsp_server.sent_frames,
                    detection_count
                )
                print("\r" + message, end="")
                udp_server.send(message)

            if frame_count % 30 == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("\n[MAIN] user stop")
    except BaseException as e:
        print("\n[MAIN] exception:")
        sys.print_exception(e)
    finally:
        print("\n[MAIN] releasing resources")
        if yolo_detector is not None:
            yolo_detector.deinit()

        if rtsp_server is not None:
            rtsp_server.stop()
        elif sensor is not None:
            sensor.stop()

        if udp_server is not None:
            udp_server.close()

        if wifi_ap is not None:
            wifi_ap.active(False)

        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)
        gc.collect()
        print("[MAIN] done")


if __name__ == "__main__":
    main()

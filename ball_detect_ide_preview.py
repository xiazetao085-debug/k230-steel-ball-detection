import gc
import os
import sys
import time

from libs.PipeLine import PipeLine
from libs.YOLO import YOLOv8


# Camera and CanMV IDE preview.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_SIZE = [FRAME_WIDTH, FRAME_HEIGHT]
MODEL_INPUT_SIZE = [320, 320]
PREVIEW_FPS = 30
DETECT_EVERY_N_FRAMES = 1

# Steel-ball YOLOv8 model.
KMODEL_PATH = "/sdcard/steel_ball_yolov8n_320.kmodel"
LABELS = ["steel_ball"]
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
MAX_BOXES = 10

# Candidate filters. These match the current RTSP version.
MIN_BOX_SIZE = 12
MAX_BOX_SIZE = 60
MIN_ASPECT_RATIO = 0.65

# Pipe region in the 640 x 480 image.
PIPE_AXIS = "x"
PIPE_ROI_X = 0
PIPE_ROI_Y = 125
PIPE_ROI_WIDTH = 500
PIPE_ROI_HEIGHT = 150
PIPE_CENTER_TOLERANCE = 65
POSITION_REVERSED = False
EXPECTED_RADIUS = 11

# Temporal tracking.
CONFIRM_DETECTIONS = 2
MISS_HOLD_DETECTIONS = 3
MAX_TRACK_JUMP = 100
FILTER_ALPHA = 0.35

# Overlay colors.
COLOR_ROI = (0, 180, 255)
COLOR_RAW = (255, 70, 70)
COLOR_CANDIDATE = (255, 220, 0)
COLOR_TRACK = (0, 255, 0)
COLOR_SEARCH = (255, 220, 0)
COLOR_TEXT = (255, 255, 255)


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
        if float(scores[index]) < CONFIDENCE_THRESHOLD:
            continue

        box = boxes[index]
        x = float(box[0])
        y = float(box[1])
        width = float(box[2])
        height = float(box[3])

        if width < MIN_BOX_SIZE or height < MIN_BOX_SIZE:
            continue
        if width > MAX_BOX_SIZE or height > MAX_BOX_SIZE:
            continue

        long_side = max(width, height)
        short_side = min(width, height)
        if short_side / long_side < MIN_ASPECT_RATIO:
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


def draw_raw_detections(osd_img, detections):
    if not detections or len(detections) < 3:
        return 0

    boxes = detections[0]
    class_ids = detections[1]
    scores = detections[2]

    for index in range(len(boxes)):
        if int(class_ids[index]) != 0:
            continue

        box = boxes[index]
        x = int(float(box[0]) + 0.5)
        y = int(float(box[1]) + 0.5)
        width = int(float(box[2]) + 0.5)
        height = int(float(box[3]) + 0.5)

        osd_img.draw_rectangle(
            x, y, width, height,
            color=COLOR_RAW,
            thickness=1
        )

        label_y = y - 18
        if label_y < 0:
            label_y = 0
        osd_img.draw_string_advanced(
            x,
            label_y,
            16,
            "{:.2f}".format(float(scores[index])),
            color=COLOR_RAW
        )

    return len(boxes)


def draw_overlay(osd_img, detections, candidates, tracker, fps):
    osd_img.clear()

    osd_img.draw_rectangle(
        PIPE_ROI_X,
        PIPE_ROI_Y,
        PIPE_ROI_WIDTH,
        PIPE_ROI_HEIGHT,
        color=COLOR_ROI,
        thickness=2
    )

    if PIPE_AXIS == "x":
        centerline = PIPE_ROI_Y + PIPE_ROI_HEIGHT // 2
        osd_img.draw_line(
            PIPE_ROI_X,
            centerline,
            PIPE_ROI_X + PIPE_ROI_WIDTH,
            centerline,
            color=COLOR_ROI,
            thickness=1
        )
    else:
        centerline = PIPE_ROI_X + PIPE_ROI_WIDTH // 2
        osd_img.draw_line(
            centerline,
            PIPE_ROI_Y,
            centerline,
            PIPE_ROI_Y + PIPE_ROI_HEIGHT,
            color=COLOR_ROI,
            thickness=1
        )

    raw_count = draw_raw_detections(osd_img, detections)

    for candidate in candidates:
        osd_img.draw_circle(
            candidate[0],
            candidate[1],
            candidate[2],
            color=COLOR_CANDIDATE,
            thickness=2
        )

    ball = tracker.get_ball()
    if ball is None:
        status_text = "BALL: {}  raw={} accepted={}".format(
            tracker.status,
            raw_count,
            len(candidates)
        )
        osd_img.draw_string_advanced(
            5, 5, 22, status_text, color=COLOR_SEARCH
        )
    else:
        axis_pixel, position = get_axis_position(ball)
        osd_img.draw_circle(
            ball[0],
            ball[1],
            ball[2] + 3,
            color=COLOR_TRACK,
            thickness=3
        )
        osd_img.draw_line(
            ball[0] - 8,
            ball[1],
            ball[0] + 8,
            ball[1],
            color=COLOR_TRACK,
            thickness=2
        )
        osd_img.draw_line(
            ball[0],
            ball[1] - 8,
            ball[0],
            ball[1] + 8,
            color=COLOR_TRACK,
            thickness=2
        )

        status_text = "BALL: {}  x={} y={} r={}".format(
            tracker.status,
            ball[0],
            ball[1],
            ball[2]
        )
        position_text = "axis_px={}  pos={:.3f}".format(
            axis_pixel,
            position
        )
        osd_img.draw_string_advanced(
            5, 5, 22, status_text, color=COLOR_TRACK
        )
        osd_img.draw_string_advanced(
            5, 31, 20, position_text, color=COLOR_TRACK
        )

    osd_img.draw_string_advanced(
        5,
        55,
        18,
        "FPS={:.1f}  red=raw yellow=accepted green=tracked".format(fps),
        color=COLOR_TEXT
    )


def print_status(tracker, candidates, raw_count, fps):
    ball = tracker.get_ball()
    if ball is None:
        print(
            "\r[BALL] status={} raw={} accepted={} fps={:.1f}    ".format(
                tracker.status,
                raw_count,
                len(candidates),
                fps
            ),
            end=""
        )
        return

    axis_pixel, position = get_axis_position(ball)
    print(
        "\r[BALL] status={} x={} y={} r={} axis={} pos={:.3f} "
        "raw={} accepted={} fps={:.1f}    ".format(
            tracker.status,
            ball[0],
            ball[1],
            ball[2],
            axis_pixel,
            position,
            raw_count,
            len(candidates),
            fps
        ),
        end=""
    )


def main():
    os.exitpoint(os.EXITPOINT_ENABLE)

    pipeline = None
    detector = None

    try:
        print("[IDE] Steel-ball online preview")
        print("[IDE] No Wi-Fi, RTSP, UDP or VLC is used")

        pipeline = PipeLine(
            rgb888p_size=FRAME_SIZE,
            display_mode="virt",
            display_size=FRAME_SIZE,
            osd_layer_num=1,
            debug_mode=0
        )
        pipeline.create(fps=PREVIEW_FPS, to_ide=True)

        detector = YOLOv8(
            task_type="detect",
            mode="video",
            kmodel_path=KMODEL_PATH,
            labels=LABELS,
            rgb888p_size=FRAME_SIZE,
            model_input_size=MODEL_INPUT_SIZE,
            display_size=pipeline.get_display_size(),
            conf_thresh=CONFIDENCE_THRESHOLD,
            nms_thresh=NMS_THRESHOLD,
            max_boxes_num=MAX_BOXES,
            debug_mode=0
        )
        detector.config_preprocess()

        print("[IDE] Model:", KMODEL_PATH)
        print("[IDE] Open CanMV Toolbox -> Preview")
        print("[IDE] ROI=(%d,%d,%d,%d), axis=%s" % (
            PIPE_ROI_X,
            PIPE_ROI_Y,
            PIPE_ROI_WIDTH,
            PIPE_ROI_HEIGHT,
            PIPE_AXIS
        ))

        tracker = BallTracker()
        clock = time.clock()
        frame_count = 0
        candidates = []
        detections = None
        raw_count = 0

        while True:
            clock.tick()
            os.exitpoint()

            frame_np = pipeline.get_frame()
            frame_count += 1

            if frame_count % DETECT_EVERY_N_FRAMES == 0:
                detections = detector.run(frame_np)
                if detections and len(detections) >= 1:
                    raw_count = len(detections[0])
                else:
                    raw_count = 0
                candidates = filter_yolo_candidates(detections)
                tracker.update(candidates)

            draw_overlay(
                pipeline.osd_img,
                detections,
                candidates,
                tracker,
                clock.fps()
            )
            pipeline.show_image()

            if frame_count % 10 == 0:
                print_status(
                    tracker,
                    candidates,
                    raw_count,
                    clock.fps()
                )

            if frame_count % 30 == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("\n[IDE] stopped by user")
    except BaseException as err:
        print("\n[IDE] exception:")
        sys.print_exception(err)
    finally:
        print("\n[IDE] releasing resources")

        if detector is not None:
            detector.deinit()

        if pipeline is not None:
            try:
                pipeline.destroy()
            except BaseException as err:
                print("[IDE] pipeline cleanup exception:")
                sys.print_exception(err)

        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)
        gc.collect()
        print("[IDE] done")


if __name__ == "__main__":
    main()

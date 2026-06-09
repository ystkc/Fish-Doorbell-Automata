# Agent instruct: 默认python未安装cv2，请使用d:\python\python.exe运行
import cv2
import httpx
import time
import numpy as np
import threading
import queue
from datetime import datetime
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Live Motion API")


import os
os.chdir(os.path.dirname(__file__))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

 # 时间戳区域坐标，在这个区域内的变化会被直接忽略
TIME_STAMP_AREA_X_MIN = 1450
TIME_STAMP_AREA_X_MAX = 1870
TIME_STAMP_AREA_X_WIDTH = TIME_STAMP_AREA_X_MAX - TIME_STAMP_AREA_X_MIN
TIME_STAMP_AREA_Y_MIN = 40
TIME_STAMP_AREA_Y_MAX = 90
TIME_STAMP_AREA_Y_HEIGHT = TIME_STAMP_AREA_Y_MAX - TIME_STAMP_AREA_Y_MIN
SHOW_TIMESTAMP_AREA = True


VALID_CONTOUR_AREA_THRESH = 150 # 有效contour的阈值(面积)
BOUNDING_BOX_MERGE_THRESH = 160 # 合并bounding box的阈值(最近边距离，h+w)
MOTION_THRESH = 50000 # 运动阈值(面积)
MOTION_SKIP_THREESH = MOTION_THRESH // 5 # 当运动面积小于这个的时候，开始增大skip_period（跳过一些帧）
MAX_SKIP_PERIOD_FRAME = 24 # 最大画面长时静止时采样周期(帧)
SKIP_PERIOD_STEP = 0.3 # # 增加skip_period的步长(帧)
FPS_DELAY = 45 # 帧率延迟(毫秒)为了避免处理太快然后卡顿

DRAW_CONTOURS = False
DRAW_CONTOURS_BOUNDING_BOX = True

SAVE_MOTION_FRAME = True
SAVE_MOTION_PERIOD = 0 # 运动周期(秒) 
SAVE_DEBUG_DRAW = True
PRINT_STATS = True

# 已知视频源固定20FPS
M3U8_URL = "https://visdeurbel.videostreams.nl/hls/visdeurbel/index.m3u8"
VIEWER_CNT_URL = "https://europe-west1-visdeurbel-454111.cloudfunctions.net/getViewerCount/wp-json/viewers/latest?cache-circumvent="
VIEWER_MULTIPLIER_URL = "https://visdeurbel.nl/wp-json/viewers/latest?lang=en&cache-circumvent="

frame_queue = queue.Queue(maxsize=10)
motion_frame_queue = queue.Queue(maxsize=10)

latest_stats = {
    "contours": 0,
    "total_area": 0,
    "motion": False,
    "web_delay": -1,
    "process_delay": -1,
    "frame_processed": 0,
    "frame_total": 0,
    "skip_period": 0,
    "skip_counter": 0
}

class UnionBoundingBox():
    '''并查集+自动合并+区域裁剪的BoundingBox'''
    def __init__(self, dist_thresh=10, prune_area=None):
        self.bounding_boxes_list = []
        self.parent_list = []
        self.dist_thresh = dist_thresh
        self.prune_area = prune_area

    def add(self, new_box):
        '''添加一个bounding box'''
        # 先判断是不是日期框内的变化
        if self.prune_area is not None and self.contain(self.prune_area, new_box):
            return
                
        self.bounding_boxes_list.append(new_box)
        new_idx = len(self.bounding_boxes_list) - 1
        self.parent_list.append(new_idx)
        
        # 不断合并所有符合条件的根框，直到没有变化
        changed = True
        while changed:
            changed = False
            # 找到所有需要与新框合并的根框
            to_merge = []
            for i, box in enumerate(self.bounding_boxes_list):
                if self.root(i) != i or i == self.root(new_idx):
                    continue
                dist = self.intersect(self.bounding_boxes_list[self.root(new_idx)], box)
                if dist < self.dist_thresh:
                    to_merge.append(i)
            
            # 合并所有找到的根框
            for i in to_merge:
                self.union(self.root(new_idx), i)
                changed = True
        
    
    def root(self, i):
        '''查找根节点'''
        while self.parent_list[i] != i:
            self.parent_list[i] = self.parent_list[self.parent_list[i]]
            i = self.parent_list[i]
        return i
    
    def union(self, i, j):
        '''合并两个集合'''
        i_root = self.root(i)
        j_root = self.root(j)
        if i_root == j_root:
            return
        self.parent_list[i_root] = j_root
        # 合并x y h w
        xi, yi, wi, hi = self.bounding_boxes_list[i_root]
        xj, yj, wj, hj = self.bounding_boxes_list[j_root]
        xnew = min(xi, xj)
        ynew = min(yi, yj) # 左上角
        xmaxnew = max(xi + wi, xj + wj)
        ymaxnew = max(yi + hi, yj + hj) # 右下角
        self.bounding_boxes_list[j_root] = (xnew, ynew, xmaxnew - xnew, ymaxnew - ynew)
    
    def get_bounding_boxes_and_area(self):
        '''获取所有bounding box'''
        result = []
        total_area = 0
        for i, box in enumerate(self.bounding_boxes_list):
            if self.root(i) == i:
                total_area += box[2] * box[3]
                result.append(box)
        return result, total_area
    
    def intersect(self, boxA, boxB):
        '''判断两个bounding box距离，横竖最近边的距离'''
        xA, yA, wA, hA = boxA
        xB, yB, wB, hB = boxB
        xmin = max(xA, xB)
        xmax = min(xA + wA, xB + wB)
        ymin = max(yA, yB)
        ymax = min(yA + hA, yB + hB)
        return xmin - xmax + ymin - ymax
    
    def contain(self, outerBox, innerBox):
        '''判断innerBox是否在outerBox内'''
        xA, yA, wA, hA = outerBox
        xB, yB, wB, hB = innerBox
        if xB >= xA and xB + wB <= xA + wA and yB >= yA and yB + hB <= yA + hA:
            return True
        return False

# -----------------------------
# 视频分析 Worker
# -----------------------------

def contour_bounding_box(contours):
    '''将contours转换为bounding box（相近的contours合并为一个bounding box）'''
    union_bounding_box = UnionBoundingBox(dist_thresh=BOUNDING_BOX_MERGE_THRESH, prune_area=(TIME_STAMP_AREA_X_MIN, TIME_STAMP_AREA_Y_MIN, TIME_STAMP_AREA_X_WIDTH, TIME_STAMP_AREA_Y_HEIGHT))
    init_bounding_boxes = []
    for c in contours:
        bounding_box = cv2.boundingRect(c)
        init_bounding_boxes.append(bounding_box)
        union_bounding_box.add(bounding_box)        
    return union_bounding_box.get_bounding_boxes_and_area(), init_bounding_boxes


def video_worker():
    global latest_stats

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "timeout;3000000|"
        "rw_timeout;3000000|"
        "reconnect;1|"
        "reconnect_streamed;1|"
        "reconnect_delay_max;2"
    )
    cap = cv2.VideoCapture(M3U8_URL,cv2.CAP_FFMPEG)
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=100, varThreshold=50, detectShadows=False
    )
    last_frame_time = int(time.time()*1000)
    last_save_time = 0
    cur_time = 0
    frame_processed = 0
    frame_total = 0
    skip_period = 0 # 画面长时静止时采样周期(帧)
    skip_counter = 0
    while True:
        # a = time.time()
        ret, frame = cap.read()
        # print(time.time() - a)
        frame_total += 1
        if not ret or (frame is None):
            # 睡眠一定时间后重试
            latest_stats.update({"web_delay": "Connection Reset"})
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(M3U8_URL)
            continue

        skip_counter += 1
        cur_time = int(time.time()*1000)
        web_delay = cur_time - last_frame_time
        last_frame_time = cur_time

        contours = merged_bounding_boxes = []
        total_area = -1
        motion = False
        frame_copy = None
        if skip_counter >= skip_period:
            frame_processed += 1
            skip_counter = 0
            frame_copy = frame.copy()

            fgmask = fgbg.apply(frame)
            _, thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # 去除太小的contours
            contours = [c for c in contours if cv2.contourArea(c) > VALID_CONTOUR_AREA_THRESH]
            (merged_bounding_boxes, total_area), init_bounding_box = contour_bounding_box(contours)

            motion = total_area > MOTION_THRESH

            if total_area < MOTION_SKIP_THREESH:
                if skip_period < MAX_SKIP_PERIOD_FRAME:
                    skip_period += SKIP_PERIOD_STEP
            else:
                skip_period = 0 # 只有被处理的帧才参与skip_period更新，被跳过的保持不变


        if DRAW_CONTOURS:
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        if DRAW_CONTOURS_BOUNDING_BOX:
            for x, y, w, h in merged_bounding_boxes:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            for x, y, w, h in init_bounding_box:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        if SHOW_TIMESTAMP_AREA:
            cv2.rectangle(frame, (TIME_STAMP_AREA_X_MIN, TIME_STAMP_AREA_Y_MIN), (TIME_STAMP_AREA_X_MAX, TIME_STAMP_AREA_Y_MAX), (0, 255, 0), 2)

        
        motion_saved = False
        if motion and SAVE_MOTION_FRAME:
            if cur_time - last_save_time >= SAVE_MOTION_PERIOD * 1000:
                last_save_time = cur_time
                motion_saved = True
                if SAVE_DEBUG_DRAW:
                    motion_frame_queue.put((frame, last_frame_time))
                else:
                    motion_frame_queue.put((frame_copy, last_frame_time))
            else:
                if PRINT_STATS:
                    print("MOTION SKIPPED")

        cv2.putText(
            frame,
            f"Motion: {motion}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if motion else (0, 0, 255),
            2,
        )

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass

        frame_queue.put(frame)
        cur_time = int(time.time()*1000)
        process_delay = cur_time - last_frame_time        
        latest_stats.update({
            "contours": len(merged_bounding_boxes),
            "total_area": int(total_area),
            "motion": motion,
            "web_delay": web_delay,
            "process_delay": process_delay,
            "timestamp": last_frame_time, # 为了保持和queue中frame的timestamp一致
            "frame_processed": frame_processed,
            "frame_total": frame_total,
            "skip_period": skip_period,
            "skip_counter": skip_counter,
        })
        if PRINT_STATS and motion_saved:
            print(latest_stats)
            print(merged_bounding_boxes)
            print(init_bounding_box)
        last_frame_time = cur_time
        sleep_time = max(0, FPS_DELAY - process_delay - web_delay)
        time.sleep(sleep_time/1000)

    cap.release()

def motion_frame_worker():
    '''将有动作的保存到文件'''
    folder_name = f"frames/{datetime.now().strftime('%Y%m%d')}"
    folder_day = datetime.now().day  
    os.makedirs(folder_name, exist_ok=True)
    while True:
        frame, timestamp = motion_frame_queue.get()
        now = datetime.now()
        if now.day != folder_day:
            folder_day = now.day
            folder_name = f"frames/{now.strftime('%Y%m%d')}"
            os.makedirs(folder_name, exist_ok=True)
        cv2.imwrite(f"{folder_name}/{now.strftime('%Y%m%d_%H%M%S')}_{timestamp}.jpg", frame)


threading.Thread(target=video_worker, daemon=True).start()
threading.Thread(target=motion_frame_worker, daemon=True).start()



# -----------------------------
# API
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <body>
        <h2>Live Motion View</h2>
        <img src="/video" style="width: 80%; height: auto;" />
        <p id="stats"></p>
        <p>Viewer Count: <span id="viewer_cnt"></span></p>

        <script>
          fetch("/viewer_cnt").then((res) => res.json()).then((j) => {
            document.getElementById("viewer_cnt").innerText = j;
          });
          setInterval(async () => {
            const r = await fetch("/stats");
            const j = await r.json();
            document.getElementById("stats").innerText =
              `Contours: ${j.contours} | Area: ${j.total_area} | Motion: ${j.motion} | Web Delay: ${j.web_delay} | Process Delay: ${j.process_delay} | Frame Processed: ${j.frame_processed} | Frame Total: ${j.frame_total} | | Skip Counter/Period: ${j.skip_counter}/${j.skip_period.toFixed(1)}`;
          }, 500);
        </script>
      </body>
    </html>
    """

@app.get("/viewer_cnt")
def viewer_cnt():
    '''获取当前viewer数量'''
    try:
        r = httpx.get(VIEWER_CNT_URL+str(int(time.time()*1000)))
        j = r.json()
        return j["latest_viewers"]
    except:
        return -1


@app.get("/video")
def video():
    def generate():
        while True:
            frame = frame_queue.get()
            _, buf = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stats")
def stats():
    return latest_stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8895)
import cv2
# 请使用d:\python\python.exe运行
import numpy as np
import time
import random

BOUNDING_BOX_MERGE_THRESH = 210 # 合并bounding box的阈值(最近边距离，h+w)

TIME_STAMP_AREA_X_MIN = 1450
TIME_STAMP_AREA_X_MAX = 1870
TIME_STAMP_AREA_X_WIDTH = TIME_STAMP_AREA_X_MAX - TIME_STAMP_AREA_X_MIN
TIME_STAMP_AREA_Y_MIN = 40
TIME_STAMP_AREA_Y_MAX = 90
TIME_STAMP_AREA_Y_HEIGHT = TIME_STAMP_AREA_Y_MAX - TIME_STAMP_AREA_Y_MIN


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


def test(box):
    union_bounding_box = UnionBoundingBox(dist_thresh=BOUNDING_BOX_MERGE_THRESH, prune_area=(TIME_STAMP_AREA_X_MIN, TIME_STAMP_AREA_Y_MIN, TIME_STAMP_AREA_X_WIDTH, TIME_STAMP_AREA_Y_HEIGHT))
    
    for b in box:
        union_bounding_box.add(b)
    result, total_area = union_bounding_box.get_bounding_boxes_and_area()
    for i, result_i in enumerate(result):
        print(i, result_i, end="")
        for j, result_j in enumerate(result):
            if i == j:
                continue
            dist = union_bounding_box.intersect(result_i, result_j)
            contain = union_bounding_box.contain(result_i, result_j)
            print(dist, contain, end=" ")
        print()

class UnionBoundingBoxNumpy():
    '''使用numpy批量计算距离的并查集BoundingBox合并类'''
    def __init__(self, dist_thresh=10, prune_area=None):
        self.dist_thresh = dist_thresh
        self.prune_area = prune_area
        self.bounding_boxes = None
        self.parent = None
    
    def _contain(self, outerBox, innerBox):
        xA, yA, wA, hA = outerBox
        xB, yB, wB, hB = innerBox
        return xB >= xA and xB + wB <= xA + wA and yB >= yA and yB + hB <= yA + hA
    
    def add_all(self, boxes):
        '''批量添加所有bounding box'''
        if self.prune_area is not None:
            boxes = [b for b in boxes if not self._contain(self.prune_area, b)]
        
        self.bounding_boxes = np.array(boxes, dtype=np.float64)
        n = len(self.bounding_boxes)
        self.parent = np.arange(n)
        
        if n == 0:
            return
        
        x = self.bounding_boxes[:, 0]
        y = self.bounding_boxes[:, 1]
        w = self.bounding_boxes[:, 2]
        h = self.bounding_boxes[:, 3]
        
        x2 = x + w
        y2 = y + h
        
        x_i = x[:, np.newaxis]
        y_i = y[:, np.newaxis]
        x2_i = x2[:, np.newaxis]
        y2_i = y2[:, np.newaxis]
        
        xmin = np.maximum(x_i, x)
        xmax = np.minimum(x2_i, x2)
        ymin = np.maximum(y_i, y)
        ymax = np.minimum(y2_i, y2)
        
        dist_matrix = (xmin - xmax) + (ymin - ymax)
        
        mask = dist_matrix < self.dist_thresh
        np.fill_diagonal(mask, False)
        
        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    self.union(i, j)
    
    def root(self, i):
        '''查找根节点（路径压缩）'''
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    
    def union(self, i, j):
        '''合并两个集合'''
        i_root = self.root(i)
        j_root = self.root(j)
        if i_root == j_root:
            return
        self.parent[i_root] = j_root
        
        xi, yi, wi, hi = self.bounding_boxes[i_root]
        xj, yj, wj, hj = self.bounding_boxes[j_root]
        xnew = min(xi, xj)
        ynew = min(yi, yj)
        xmaxnew = max(xi + wi, xj + wj)
        ymaxnew = max(yi + hi, yj + hj)
        self.bounding_boxes[j_root] = [xnew, ynew, xmaxnew - xnew, ymaxnew - ynew]
    
    def get_bounding_boxes_and_area(self):
        '''获取所有合并后的bounding box'''
        if self.bounding_boxes is None:
            return [], 0
        
        result = []
        total_area = 0
        n = len(self.bounding_boxes)
        
        for i in range(n):
            if self.root(i) == i:
                box = tuple(self.bounding_boxes[i].astype(int))
                total_area += box[2] * box[3]
                result.append(box)
        
        return result, total_area


def generate_random_contours(num_boxes, max_x=1920, max_y=1080, max_w=100, max_h=100):
    '''生成随机bounding box数据'''
    contours = []
    for _ in range(num_boxes):
        x = random.randint(0, max_x - 10)
        y = random.randint(0, max_y - 10)
        w = random.randint(10, max_w)
        h = random.randint(10, max_h)
        contours.append((x, y, w, h))
    return contours


def performance_test(num_trials=100, max_boxes=100):
    '''测试两个类的性能'''
    print("Performance Test Results:")
    print("-" * 60)
    
    for num_boxes in [10, 30, 50, 70, 100]:
        total_time_original = 0.0
        total_time_numpy = 0.0
        
        for _ in range(num_trials):
            contours = generate_random_contours(num_boxes)
            
            start = time.time()
            ubb = UnionBoundingBox(dist_thresh=BOUNDING_BOX_MERGE_THRESH)
            for box in contours:
                ubb.add(box)
            ubb.get_bounding_boxes_and_area()
            total_time_original += time.time() - start
            
            start = time.time()
            ubb_np = UnionBoundingBoxNumpy(dist_thresh=BOUNDING_BOX_MERGE_THRESH)
            ubb_np.add_all(contours)
            ubb_np.get_bounding_boxes_and_area()
            total_time_numpy += time.time() - start
        
        avg_original = (total_time_original / num_trials) * 1000
        avg_numpy = (total_time_numpy / num_trials) * 1000
        
        print(f"Boxes: {num_boxes:3d} | Original: {avg_original:6.2f} ms | NumPy: {avg_numpy:6.2f} ms | Speedup: {avg_original/avg_numpy:.2f}x")


contours = [(847, 1070, 34, 10), (1817, 1048, 16, 32), (161, 1045, 34, 24), (1804, 1003, 48, 54), (1663, 986, 41, 33), (98, 975, 34, 17), (1662, 967, 35, 32), (63, 963, 23, 33), (83, 961, 39, 31), (391, 943, 41, 44), (160, 934, 32, 32), (198, 919, 58, 40), (378, 911, 63, 60), (131, 909, 36, 27), (42, 909, 29, 48), (187, 906, 36, 30), (70, 896, 59, 67), (373, 895, 41, 55), (775, 891, 30, 15), (1532, 889, 32, 29), (536, 876, 34, 24), (1303, 866, 32, 17), (201, 864, 158, 119), (80, 861, 19, 21), (724, 859, 42, 13), (129, 856, 46, 48), (163, 855, 40, 46), (157, 854, 78, 94), (232, 851, 36, 19), (221, 824, 42, 20), (1333, 823, 23, 16), (906, 819, 43, 16), (287, 819, 61, 61), (1189, 815, 39, 17), (347, 807, 22, 20), (1413, 806, 39, 36), (1249, 803, 38, 26), (261, 794, 30, 44), (1389, 784, 21, 20), (122, 784, 20, 42), (904, 783, 31, 28), (1161, 781, 25, 17), (320, 766, 67, 47), (957, 761, 22, 28), (880, 759, 37, 28), (1314, 749, 74, 46), (1117, 748, 52, 30), (220, 745, 99, 69), (1078, 725, 31, 34), (846, 702, 61, 37), (253, 679, 353, 241), (914, 677, 24, 14), (611, 662, 33, 19), (794, 655, 30, 17), (1177, 651, 14, 21), (1066, 630, 624, 402), (645, 618, 102, 53), (382, 617, 429, 276), (887, 599, 57, 52), (1003, 569, 27, 21), (887, 534, 79, 46), (946, 527, 56, 40), (752, 525, 166, 103), (296, 0, 26, 33), (5, 0, 292, 192)]

print("Testing original UnionBoundingBox:")
test(contours)

print("\nTesting UnionBoundingBoxNumpy:")
def test_numpy(box):
    union_bounding_box = UnionBoundingBoxNumpy(dist_thresh=BOUNDING_BOX_MERGE_THRESH, prune_area=(TIME_STAMP_AREA_X_MIN, TIME_STAMP_AREA_Y_MIN, TIME_STAMP_AREA_X_WIDTH, TIME_STAMP_AREA_Y_HEIGHT))
    union_bounding_box.add_all(box)
    result, total_area = union_bounding_box.get_bounding_boxes_and_area()
    for i, result_i in enumerate(result):
        print(i, result_i, end="")
        for j, result_j in enumerate(result):
            if i == j:
                continue
            dist = union_bounding_box._intersect(result_i, result_j) if hasattr(union_bounding_box, '_intersect') else 0
            contain = union_bounding_box._contain(result_i, result_j)
            print(dist, contain, end=" ")
        print()

test_numpy(contours)

print("\n" + "="*60)
performance_test()
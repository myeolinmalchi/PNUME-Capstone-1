from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from cv2.typing import MatLike

from utils.wrappers import RKNNWrapper


def init_rknns(model_path: str, N: int):
    rknns = []
    for _ in range(N):
        rknns.append(RKNNWrapper(model_path))
    return rknns

        
class RKNNPoolExecutor():
    def __init__(self, model_path: str, N: int, func):
        self.queue = Queue()
        self.rknns = init_rknns(model_path, N)
        self.pool = ThreadPoolExecutor(max_workers=N)
        self.func = func
        self.num = 0
        self.N = N

    def put(self, frame: MatLike):
        future = self.pool.submit(self.func, self.rknns[self.num % self.N], frame)
        self.queue.put(future)
        self.num += 1

    def get(self):
        if self.queue.empty():
            return None, None
        future = self.queue.get()
        result = future.result()
        return result

    def release(self):
        self.pool.shutdown()
        for rknn in self.rknns:
            rknn.release()

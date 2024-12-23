from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from cv2.typing import MatLike

from utils.wrappers import RKNNWrapper

class RKNNPoolExecutor():
    def __init__(self, model_path: str, N: int, func):
        self.queue = Queue()
        self.rknn = RKNNWrapper(model_path)
        self.pool = ThreadPoolExecutor(max_workers=N)
        self.func = func

    def put(self, frame: MatLike):
        future = self.pool.submit(self.func, self.rknn, frame)
        self.queue.put(future)

    def get(self):
        if self.queue.empty():
            return [], []
        future = self.queue.get()
        result = future.result()
        return result

    def release(self):
        self.pool.shutdown()
        self.rknn.release

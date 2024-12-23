from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import threading

from cv2.typing import MatLike

from m1s.utils.wrappers import RKNNWrapper

class RKNNPoolExecutor():
    def __init__(self, model_path: str, N: int, func):
        self.queue = Queue()
        self.rknn = RKNNWrapper(model_path)
        self.semaphore = threading.Semaphore(value=N)
        self.pool = ThreadPoolExecutor(max_workers=N)
        self.func = func

    def put(self, frame: MatLike):
        self.semaphore.acquire()
        future = self.pool.submit(self.func, self.rknn, frame)
        self.queue.put(future)

    def get(self):
        if self.queue.empty():
            return [], []
        result = self.queue.get()
        return result

    def release(self):
        self.pool.shutdown()
        self.rknn.release

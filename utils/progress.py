from rich.progress import Progress, SpinnerColumn, TextColumn
from threading import Lock


class ProgressBarMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class ProgressBarBase(metaclass=ProgressBarMeta):
    progress = None
    isStarted = False

    def __init__(self):
        self.progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                                 transient=True)

    def start(self):
        if not self.isStarted:
            self.progress.start()
            self.isStarted = True

    def stop(self):
        if self.isStarted:
            self.progress.stop()
            self.isStarted = False


class ProgressBarController(ProgressBarBase):
    def __init__(self):
        super().__init__()
        self.task_ids = []

    def new(self, message):
        self.start()
        task_id = self.progress.add_task(description=message, total=None)
        self.task_ids.append(task_id)

    def terminate(self):
        last_id = self.task_ids.pop()
        self.progress.remove_task(last_id)

    def __del__(self):
        del self
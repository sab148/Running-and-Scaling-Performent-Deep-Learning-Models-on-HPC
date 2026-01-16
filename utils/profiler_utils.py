import torch
import time

class ProfilerSection:
    """
    Context manager for profiling sections of PyTorch code using NVTX ranges.

    Attributes:
        name (str): The name of the section for profiling.
        profile (bool): Whether profiling is enabled.
    """

    def __init__(self, name: str, profile: bool = False):
        """
        Initialize a ProfilerSection.

        Args:
            name (str): Name of the section.
            profile (bool): Whether to enable profiling (default: False).
        """
        self.profile = profile
        self.name = name

    def __enter__(self):
        """
        Enter the context.

        If profiling is enabled, push a new NVTX range with the given name.
        NVTX ranges can be visualized in NVIDIA profiling tools to see timing.
        """
        if self.profile:
            torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context.

        If profiling is enabled, pop the NVTX range started in __enter__.
        """
        if self.profile:
            torch.cuda.nvtx.range_pop()

class ExecutionTimer(object):
    def __init__(self, name: str, profile: bool = False):
        self._name = name
        self._profile = profile

    def __enter__(self):
        torch.cuda.nvtx.range_push(self._name)
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        torch.cuda.nvtx.range_pop()
        self._stop_time = time.time()

    def start(self):
        self._start_time = time.time()

    def time_elapsed(self) -> float:
        if not hasattr(self, "_stop_time"):
            return time.time() - self._start_time

        return self._stop_time - self._start_time

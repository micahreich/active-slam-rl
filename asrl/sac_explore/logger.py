import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class Logger:
    def __init__(self, log_base_dir='runs', name='exp'):
        """
        Args:
            log_base_dir (str): Root directory where logs will be saved.
            name (str): Descriptive name for the experiment.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_base_dir, "runs", f"{name}_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        print(f"[Logger] Logging to {self.log_dir}")

    def log(self, tag, value, step):
        """
        Logs a scalar value to TensorBoard.

        Args:
            tag (str): Tag under which to log the value.
            value (float): Value to log.
            step (int): Training step.
        """
        self.writer.add_scalar(tag, value, step)

    def flush(self):
        """Ensure logs are written to disk."""
        self.writer.flush()

    def close(self):
        """Close the writer safely."""
        self.writer.close()
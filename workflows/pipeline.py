import os

# Suppress TensorFlow GPU/Metal init on macOS (prevents mutex lock stall)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"

# Redirect C-level stderr (fd 2) to /dev/null during import so that
# RAW: mutex.cc messages — which bypass Python logging — are suppressed.
_devnull = os.open(os.devnull, os.O_WRONLY)
_stderr_fd = os.dup(2)
os.dup2(_devnull, 2)
os.close(_devnull)

from lk_leopards import (
    LeopardAI,
    ReadMeBuilder,
)  # noqa: E402 (imports TF/DeepFace)

# Restore stderr so Rich console output works normally
os.dup2(_stderr_fd, 2)
os.close(_stderr_fd)

if __name__ == "__main__":
    LeopardAI().build_fingerprints()
    ReadMeBuilder().write()

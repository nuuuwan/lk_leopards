import argparse
import os

from lk_leopards import LeopardAI  # noqa: E402
from lk_leopards.ReadMeBuilder import ReadMeBuilder  # noqa: E402
from lk_leopards.SimilarityBuilder import SimilarityBuilder  # noqa: E402

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


# Restore stderr so Rich console output works normally
os.dup2(_stderr_fd, 2)
os.close(_stderr_fd)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N images.",
    )
    args = parser.parse_args()
    ai = LeopardAI()
    ai.build_face_detected(max_images=args.n)
    ai.build_faces_from_detected(max_images=args.n)
    ai.build_fingerprints()
    SimilarityBuilder().write()
    ReadMeBuilder().write()

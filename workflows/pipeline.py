import os
import sys

from lk_leopards.LeopardAI import LeopardAI
from lk_leopards.ReadMeBuilder import ReadMeBuilder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


if __name__ == "__main__":
    LeopardAI().build_fingerprints()
    ReadMeBuilder().write()

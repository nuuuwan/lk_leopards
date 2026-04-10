import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lk_leopards.ReadMeBuilder import ReadMeBuilder

if __name__ == "__main__":
    ReadMeBuilder().write()


if __name__ == "__main__":
    ReadMeBuilder().write()

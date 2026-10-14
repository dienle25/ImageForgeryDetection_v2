"""Local CPU image forensics: manipulation detection/localization with explicit scope and provenance."""
import os as _os

_os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # silence OpenCV DNN backend notices in the console

__version__ = "2.0.0"

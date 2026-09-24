"""Compatibility shim: the Flask demo was replaced by the FastAPI app.

    python app.py        # same as: netra_ocr serve --host 0.0.0.0 --port 5000
"""

from netra_ocr.server.__main__ import main

if __name__ == "__main__":
    main(["--host", "0.0.0.0", "--port", "5000"])

"""``python -m netra_ocr.server`` / ``netra_ocr serve``: run the web app + API."""

import argparse
import logging
import os


def main(argv=None):
    parser = argparse.ArgumentParser(prog="netra_ocr serve", description="Run the Netra OCR web app and REST API.")
    parser.add_argument("--host", default=os.environ.get("NETRA_HOST", "127.0.0.1"),
                        help="Interface to bind (default 127.0.0.1; use 0.0.0.0 to allow other machines)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("NETRA_PORT", 8000)))
    parser.add_argument("--data-dir", help="Where jobs are stored (default ~/.cache/netra-ocr/jobs)")
    args = parser.parse_args(argv)
    if args.data_dir:
        os.environ["NETRA_DATA_DIR"] = args.data_dir

    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit("The web app needs the 'server' extra: pip install \"netra-ocr[server]\"") from e
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(f"Netra OCR: open http://{'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}")
    if os.path.exists("/.dockerenv"):
        # Inside a container the URL above only works if the port was published.
        print(f"Running in Docker: start the container with -p {args.port}:{args.port} "
              f"(Docker Desktop: Optional settings → Host port {args.port}), or the page won't open.")
    # A single process: the models are loaded once and one worker thread runs
    # every OCR job (see server/jobs.py).
    uvicorn.run("netra_ocr.server.app:app", host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()

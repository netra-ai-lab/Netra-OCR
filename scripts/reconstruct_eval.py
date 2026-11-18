"""See how well the original-layout Word export rebuilds each page.

For every input (default: test_images/) it runs the layout pipeline with
``reconstruct=True`` and writes, under --out:

    <name>.docx            the rebuilt document
    <name>.sheet.png       scan | plan preview | LibreOffice render, side by side
    <name>.json            the measured document (styles, overlays)

and prints an overlap score per page: how much of the scan's ink lands
where the rebuilt page has ink, on a blurred mask (1.0 = identical
layout). The LibreOffice render shows what Word-compatible software makes
of the .docx (wrapping, fonts, floats); the preview shows the plan alone.
LibreOffice is found on PATH, under /opt/libreoffice*, or via $SOFFICE.

    python scripts/reconstruct_eval.py                     # all of test_images/
    python scripts/reconstruct_eval.py letter.jpg --out /tmp/rebuild
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from netra_ocr.exporters import export  # noqa: E402
from netra_ocr.ocr_engine import KhmerOCRPipeline  # noqa: E402
from netra_ocr.reconstruct import plan as PL, preview as PV  # noqa: E402

PX_PER_PT = 1.5


def find_soffice():
    for cand in [os.environ.get("SOFFICE"), shutil.which("soffice"), shutil.which("libreoffice"),
                 *sorted(glob.glob("/opt/libreoffice*/program/soffice"))]:
        if cand and os.path.exists(cand):
            return cand
    return None


def docx_to_pages(soffice, docx_path):
    """Render a .docx to page images with LibreOffice (None if unavailable)."""
    if not soffice:
        return None
    import pypdfium2 as pdfium
    with tempfile.TemporaryDirectory() as tmp:
        profile = "file://" + os.path.join(tmp, "profile")
        subprocess.run([soffice, "--headless", f"-env:UserInstallation={profile}",
                        "--convert-to", "pdf", "--outdir", tmp, docx_path],
                       check=True, capture_output=True, timeout=300)
        pdf_path = os.path.join(tmp, os.path.splitext(os.path.basename(docx_path))[0] + ".pdf")
        pdf = pdfium.PdfDocument(pdf_path)
        pages = [pdf[i].render(scale=PX_PER_PT).to_pil().convert("RGB") for i in range(len(pdf))]
        pdf.close()
        return pages


def scan_on_sheet(doc, page_index, size):
    """The scan, placed on the sheet the way the plan places it."""
    page = doc.pages[page_index]
    scan = doc.assets.get(page["image"]).convert("RGB")
    k = PX_PER_PT * 72 / page["dpi"]
    sheet = Image.new("RGB", size, "white")
    ox = int(max(0.0, (page["size_pt"][0] - page["width"] * 72 / page["dpi"]) / 2) * PX_PER_PT)
    sheet.paste(scan.resize((max(1, int(scan.width * k)), max(1, int(scan.height * k)))), (ox, 0))
    return sheet


def ink_overlap(a: Image.Image, b: Image.Image) -> float:
    """IoU of blurred ink masks: forgiving of a pixel or two, not of a line."""
    from netra_ocr.imaging import otsu_threshold

    def mask(im):
        # Ink against the paper's own tone (a grey photo isn't all ink).
        g = np.asarray(im.convert("L"))
        ink = Image.fromarray(((g < min(otsu_threshold(g), 200)) * 255).astype(np.uint8))
        return np.asarray(ink.filter(ImageFilter.GaussianBlur(3))) > 25
    ma, mb = mask(a), mask(b.resize(a.size))
    union = (ma | mb).sum()
    return float((ma & mb).sum() / union) if union else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="images or PDFs (default: test_images/*)")
    ap.add_argument("--out", default=os.path.join(ROOT, "reconstruct_out"))
    ap.add_argument("--max-pages", type=int, default=3)
    ap.add_argument("--fonts", default="embed", choices=["embed", "system"])
    args = ap.parse_args()

    inputs = args.inputs or sorted(glob.glob(os.path.join(ROOT, "test_images", "*")))
    os.makedirs(args.out, exist_ok=True)
    soffice = find_soffice()
    if not soffice:
        print("LibreOffice not found: sheets show the plan preview only.")
    pipeline = KhmerOCRPipeline(layout=True, reconstruct=True)

    for path in inputs:
        name = os.path.basename(path)
        t0 = time.time()
        doc = pipeline.process_document(path, max_pages=args.max_pages)
        t_ocr = time.time() - t0
        data = export(doc, "docx", fonts=args.fonts)[0]
        docx_path = os.path.join(args.out, name + ".docx")
        with open(docx_path, "wb") as f:
            f.write(data)
        with open(os.path.join(args.out, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, ensure_ascii=False, indent=1)

        rendered = docx_to_pages(soffice, docx_path)
        plans = PL.build_document(doc)
        scores = []
        for i, plan in enumerate(plans):
            preview = PV.render(plan, doc.assets, PX_PER_PT)
            scan = scan_on_sheet(doc, i, preview.size)
            panels = [scan, preview]
            score = ink_overlap(scan, preview)
            if rendered is not None and i < len(rendered):
                panels.append(rendered[i].resize(preview.size))
                score = ink_overlap(scan, rendered[i])
            scores.append(score)
            sheet = Image.new("RGB", (sum(p.width for p in panels) + 10 * (len(panels) - 1),
                                      preview.height), (128, 128, 128))
            x = 0
            for p in panels:
                sheet.paste(p, (x, 0))
                x += p.width + 10
            suffix = f".p{i}" if len(plans) > 1 else ""
            sheet.save(os.path.join(args.out, f"{name}{suffix}.sheet.png"))
        extra = f", {len(rendered)} rendered page(s)" if rendered is not None else ""
        if rendered is not None and len(rendered) != len(plans):
            extra += f" -- PAGE COUNT DIFFERS ({len(plans)} planned)"
        print(f"{name:40s} ocr+measure {t_ocr:5.1f}s  overlap "
              f"{' '.join(f'{s:.2f}' for s in scores)}{extra}")
    print(f"\nOutputs in {args.out}")


if __name__ == "__main__":
    main()

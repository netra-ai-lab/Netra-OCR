from netra_ocr.ocr_engine import KhmerOCRPipeline

# Initialize the pipeline
pipeline = KhmerOCRPipeline(detector="yolo", conf=0.4)

# Process an image — returns plain text and writes the output file
result_text = pipeline.process_image(
    image_path="./test_images/test_image.jpg",
    output_path="document.docx",   # Extension determines format
    batch_size=16,
    save_debug=False,
)

print(result_text)
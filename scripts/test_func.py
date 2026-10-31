# from recognition.recognize_text import recognize

# # Basic usage (uses defaults defined in the file)
# text = recognize("test_image_5.png")
# print(text)

# from detection.detector import LayoutInference

# engine = LayoutInference()
# # This will return the list of bounding boxes
# engine.run("./detection/mef_2.tiff")
# # engine.run("./detection/goc_4.tiff") 
# # engine.run("./detection/pp_gov_4.tiff")


from ocr_engine import KhmerOCRPipeline

# 1. Initialize once (loads models)
pipeline = KhmerOCRPipeline()

# 2. Process multiple images
text = pipeline.process_image(
    image_path="./assets/khmer_document_4.jpg", 
    padding=5, 
    output_path="khmer_document_4.txt",
)

print(f"OCR Result: {text}")


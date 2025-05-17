import io
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw
import torch # Still useful for device selection

# Import your YOLOv8 model class (if you need its structure)
# Or directly use ultralytics.YOLO if you are just loading a .pt file
from ultralytics import YOLO

# --- Configuration ---
# Path to trained YOLOv8 model

MODEL_PATH = "artifacts/models/yolo_gun_lr0.0001_e30_best.pt" # 

# --- Load Model ---
# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    print("Please ensure your DVC pipeline has run successfully and the model is in the correct location.")
    # You might want to raise an exception or exit if the model isn't found
    # For now, we'll let it try to load and fail if not present.
    model = None
else:
    try:
        # Load your trained YOLOv8 model
        model = YOLO(MODEL_PATH)
        # No explicit model.eval() is typically needed for Ultralytics YOLO predict,
        # and device handling is often managed by the predict method or model initialization.
        # You can set a default device for the model if desired:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device) # This might be needed if you want to force a device globally
        print(f"Successfully loaded YOLOv8 model from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading YOLOv8 model from {MODEL_PATH}: {e}")
        model = None # Ensure model is None if loading fails

app = FastAPI()

def predict_and_draw_yolo(image: Image.Image):
    """
    Performs prediction with YOLOv8 and draws bounding boxes.
    """
    if model is None:
        raise RuntimeError("YOLOv8 model is not loaded. Cannot perform prediction.")

    # Perform inference
    # YOLO's predict method can take various sources, including PIL images
    # It handles preprocessing internally.
    # You can specify device, confidence threshold, etc., in the predict call.
    results = model.predict(source=image, conf=0.5, device='cpu') # Using CPU for broader compatibility, change if GPU needed

    # `results` is a list of Results objects. For a single image, we take the first one.
    if not results or len(results) == 0:
        return image.convert("RGB") # Return original image if no results

    result = results[0] # Get the Results object for the first (and only) image


    img_rgb = image.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)
    
    # Get boxes, scores, and labels
    boxes = result.boxes.xyxy.cpu().numpy()  # xyxy format
    scores = result.boxes.conf.cpu().numpy()
    labels_indices = result.boxes.cls.cpu().numpy()
    class_names = result.names # Dictionary of class index to class name
    
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        label_idx = int(labels_indices[i])
        class_name = class_names.get(label_idx, "Unknown")
    
        # You can set a score threshold here if not done in model.predict()
        # if score > 0.7: # Example threshold
        x_min, y_min, x_max, y_max = box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_min, y_min - 10), f"{class_name} {score:.2f}", fill="red") # Draw label and score
    img_with_detections_pil = img_rgb

    return img_with_detections_pil


@app.get("/")
def read_root():
    return {"message": "Welcome to the Guns Object Detection API (YOLOv8)"}


@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Please check server logs."}
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        output_image_pil = predict_and_draw_yolo(image)

        img_byte_arr = io.BytesIO()
        output_image_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0) # Move cursor to the beginning of the buffer

        return StreamingResponse(img_byte_arr, media_type="image/png")
    except RuntimeError as e: # Catch runtime error if model isn't loaded
        return {"error": str(e)}
    except Exception as e:
        print(f"Error during prediction: {e}") # Log the error server-side
        return {"error": f"An error occurred during prediction: {e}"}

# To run this app (save as main.py or similar):
# uvicorn main:app --reload
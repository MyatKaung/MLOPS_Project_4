from ultralytics import YOLO
# Assuming src.logger and src.custom_exception are in your project's src directory
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class YOLOv8Model:
    def __init__(self, model_variant: str = 'yolov8n.pt'):
        """
        Initializes the YOLOv8 model.

        Args:
            model_variant (str): The YOLOv8 model variant to load (e.g., 'yolov8n.pt', 'yolov8s.pt').
                                 This can also be a path to a custom .pt file.
        """
        try:
            self.model_variant = model_variant
            self.model = YOLO(self.model_variant)
            logger.info(f"YOLOv8 model initialized with variant: {self.model_variant}")
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8 model: {e}")
            raise CustomException(f"Failed to initialize YOLOv8 model with variant {self.model_variant}", e)

    def get_model(self):
        """
        Returns the loaded YOLO model.
        """
        return self.model

if __name__ == "__main__":
    try:
        yolo_model_wrapper = YOLOv8Model(model_variant='yolov8n.pt') # Loads pre-trained yolov8n
        model = yolo_model_wrapper.get_model()
        print(f"Successfully loaded YOLO model: {type(model)}")


    except CustomException as e:
        print(f"A custom error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
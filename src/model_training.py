import os
import torch
import shutil
import numpy as np
import cv2

# first thing: enable Ultralytics’ MLflow integration
from ultralytics import settings
settings.update({"mlflow": True})  # turns on the MLflow callback :contentReference[oaicite:0]{index=0}

from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_processing import prepare_yolo_dataset
from src.model_architecture import YOLOv8Model

logger = get_logger(__name__)

# Define global paths
MODEL_SAVE_DIR = "artifacts/models"
YOLO_RUNS_OUTPUT_DIR = "artifacts/yolo_training_runs"
YOLO_DATASET_DIR = "artifacts/gun_dataset_yolo"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(YOLO_RUNS_OUTPUT_DIR, exist_ok=True)
os.makedirs(YOLO_DATASET_DIR, exist_ok=True)

class YOLOModelTrainer:
    def __init__(self,
                 original_dataset_path: str,
                 model_variant: str = 'yolov8n.pt',
                 class_names: list = None,
                 epochs: int = 30,
                 batch_size: int = 4,
                 img_size: int = 640,
                 learning_rate: float = 1e-4,
                 optimizer: str = 'Adam',
                 weight_decay: float = 5e-4,
                 device: str = None,
                 project_name: str = None,
                 run_name: str = "yolo_run",
                 mlflow_experiment_name: str = "YOLO_Gun_Detection"
                 ):
        if class_names is None:
            class_names = ['gun']
        self.original_dataset_path = original_dataset_path
        self.yolo_dataset_base_dir = YOLO_DATASET_DIR
        
        self.model_variant = model_variant
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.project_name = project_name or YOLO_RUNS_OUTPUT_DIR
        self.run_name = run_name
        
        # Ultralytics callback will use this name/project to tag the run
        os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_experiment_name
        os.environ["MLFLOW_RUN_NAME"] = run_name
        
        # initialize the YOLOv8 model wrapper
        self.yolo_model_wrapper = YOLOv8Model(model_variant=self.model_variant)
        self.model = self.yolo_model_wrapper.get_model()
        
        logger.info(f"YOLOModelTrainer initialized (device={self.device}, experiment={mlflow_experiment_name}, run={run_name})")

    def prepare_data(self, train_ratio: float = 0.8):
        """Prepares the dataset and sets data_yaml_path."""
        try:
            logger.info(f"Preparing YOLO dataset (train_ratio={train_ratio})…")
            self.data_yaml_path = prepare_yolo_dataset(
                original_root=self.original_dataset_path,
                yolo_base_dir=self.yolo_dataset_base_dir,
                train_ratio=train_ratio,
                class_names=self.class_names
            )
            logger.info(f"Dataset ready: {self.data_yaml_path}")
        except Exception as e:
            logger.error("Data preparation failed", exc_info=True)
            raise CustomException("Failed during data preparation", e)

    def train(self):
        """Runs YOLO training with Ultralytics’ MLflow callback logging everything for you."""
        if not hasattr(self, 'data_yaml_path') or not os.path.exists(self.data_yaml_path):
            raise CustomException("Data YAML not found—call prepare_data() first.", None)

        try:
            logger.info("Starting YOLO training (with integrated MLflow callback)…")
            results = self.model.train(
                data=self.data_yaml_path,
                epochs=self.epochs,
                imgsz=self.img_size,
                batch=self.batch_size,
                workers=2,
                optimizer=self.optimizer,
                lr0=self.learning_rate,
                weight_decay=self.weight_decay,
                device=self.device,
                project=self.project_name,  # local save dir base
                name=self.run_name,         # sub-folder for this run
                exist_ok=True
            )
            logger.info("Training complete.")
            
            # Copy the final checkpoints locally for your own convenience
            ckpt_dir = os.path.join(results.save_dir, "weights")
            for suffix in ("last.pt", "best.pt"):
                src = os.path.join(ckpt_dir, suffix)
                dst = os.path.join(MODEL_SAVE_DIR, f"{self.run_name}_{suffix}")
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    logger.info(f"Copied checkpoint to {dst}")
            
            return results

        except Exception as e:
            logger.error("Training failed", exc_info=True)
            raise CustomException("Failed during YOLO training", e)


if __name__ == "__main__":
    raw_data = "artifacts/raw"
    # Ensure minimal dummy data exists:
    img_dir = os.path.join(raw_data, "Images")
    lbl_dir = os.path.join(raw_data, "Labels")
    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir) or not os.listdir(img_dir):
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        dummy = np.zeros((100, 100, 3), np.uint8)
        cv2.imwrite(os.path.join(img_dir, "dummy.jpeg"), dummy)
        with open(os.path.join(lbl_dir, "dummy.txt"), "w") as f:
            f.write("0 0.5 0.5 0.5 0.5")
    current_epochs =30
    current_learning_rate= 1e-4
    dynamic_run_name = f"yolo_gun_lr{current_learning_rate}_e{current_epochs}"


    trainer = YOLOModelTrainer(
        original_dataset_path=raw_data,
        model_variant='yolov8n.pt',
        class_names=['gun'],
        epochs=current_epochs,
        batch_size=4,
        img_size=640,
        learning_rate=current_learning_rate,
        optimizer='Adam',
        project_name=YOLO_RUNS_OUTPUT_DIR,
        run_name=dynamic_run_name ,
        mlflow_experiment_name="YOLOv8_Gun_Object_Detection"
    )
    trainer.prepare_data(train_ratio=0.8)
    results = trainer.train()
    print(f"Local outputs in: {results.save_dir}")
    print(f"Check MLflow UI (experiment ‘{os.environ['MLFLOW_EXPERIMENT_NAME']}’) for all metrics, parameters, and artifacts.")

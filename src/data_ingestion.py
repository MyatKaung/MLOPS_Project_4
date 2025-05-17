import os
import kagglehub
from src.logger import get_logger 
from src.custom_exception import CustomException
from config.data_ingestion_config import *
import shutil 
import zipfile

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, dataset_name: str, target_dir: str):
        self.dataset_name = dataset_name  # Format: "username/dataset-name"
        self.target_dir = target_dir
        self.raw_dir = os.path.join(target_dir, "raw")
        self.extracted_dir = os.path.join(target_dir, "extracted")

    def create_raw_dir(self):
        """Create raw data directory if it doesn't exist."""
        os.makedirs(self.raw_dir, exist_ok=True)
        logger.info(f"Raw directory created at {self.raw_dir}")

    def download_dataset(self):
        try:
            path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Downloaded dataset to: {path}")
            logger.info(f"Contents of dataset dir: {os.listdir(path)}")

            allowed_dirs = {'images', 'labels'}
            for file in os.listdir(path):
                src = os.path.join(path, file)
                if file.lower() in allowed_dirs:
                    dst = os.path.join(self.raw_dir, file)
                    shutil.move(src, dst)
                    logger.info(f"Copied: {file} → {dst}")

        except Exception as e:
            raise CustomException("Error while downloading dataset", e)

    def extract_zip_file(self, zip_path: str):
        """Extract a ZIP file to the extracted directory."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extracted_dir)
            logger.info(f"Extracted {zip_path} to {self.extracted_dir}")
        except Exception as e:
            logger.error(f"Error extracting ZIP file: {e}")
            raise

    def extract_images_and_labels(self):
        """Extract images and labels from downloaded archives."""
        for filename in os.listdir(self.raw_dir):
            if filename.endswith(".zip"):
                zip_path = os.path.join(self.raw_dir, filename)
                self.extract_zip_file(zip_path)

    def run(self):
        """Run the full data ingestion pipeline."""
        self.create_raw_dir()
        self.download_dataset()
        self.extract_images_and_labels()
        logger.info("Data ingestion pipeline completed successfully.")

if __name__ == "__main__":


    data_ingestion = DataIngestion(dataset_name=DATASET_NAME, target_dir=TARGET_DIR)
    data_ingestion.run()


                
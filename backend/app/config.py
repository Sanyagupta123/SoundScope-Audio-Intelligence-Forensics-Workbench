import os

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_DIMENSION = 8192
MODEL_NAME = os.getenv("SCENEMIND_MODEL", "yolo11n.pt")
CORS_ORIGINS = os.getenv("SCENEMIND_CORS_ORIGINS", "http://localhost:5173").split(",")

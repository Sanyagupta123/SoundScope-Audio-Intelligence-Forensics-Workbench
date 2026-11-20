import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR=Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR/".env")
STORAGE_DIR=Path(os.getenv("SOUNDSCOPE_STORAGE",str(BASE_DIR/"uploads"))).resolve()
DATABASE_PATH=Path(os.getenv("SOUNDSCOPE_DATABASE",str(BASE_DIR/"soundscope.db"))).resolve()
MAX_FILE_BYTES=int(os.getenv("SOUNDSCOPE_MAX_FILE_BYTES",25*1024*1024))
CORS_ORIGINS=os.getenv("SOUNDSCOPE_CORS_ORIGINS","http://localhost:5173").split(",")
MODEL_ID=os.getenv("SOUNDSCOPE_MODEL","MIT/ast-finetuned-audioset-10-10-0.4593")

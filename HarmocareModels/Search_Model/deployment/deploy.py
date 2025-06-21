import os
import certifi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
from pathlib import Path
from huggingface_hub import HfApi, login
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Files to skip for security
SKIP_FILES = {".env.deployment"}
SKIP_FOLDERS = {"venv", "__pycache__"}

def upload_folder_or_files(api, folder_path, repo_id, base_path):
    """Recursively upload a folder; if it fails, upload files inside it individually."""
    try:
        rel_path = folder_path.relative_to(base_path)
        logger.info(f"Uploading folder: {rel_path}")
        api.upload_folder(
            folder_path=str(folder_path),
            path_in_repo=str(rel_path),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=[
                "*.pyc", "__pycache__", "*.log",
                ".git*", "*.tmp", "*.temp", "*.bak"
            ],
            commit_message=f"Upload folder {rel_path}",
            create_pr=False
        )
        logger.info(f"Uploaded folder: {rel_path}")
        time.sleep(60)
    except Exception as e:
        logger.error(f"Failed to upload folder {rel_path}: {str(e)}")
        # Try uploading files inside the folder individually
        for file in folder_path.iterdir():
            if file.is_file():
                if file.name in SKIP_FILES:
                    logger.info(f"Skipping secret file: {file.name}")
                    continue
                try:
                    file_rel_path = file.relative_to(base_path)
                    logger.info(f"Uploading file: {file_rel_path}")
                    api.upload_file(
                        path_or_fileobj=str(file),
                        path_in_repo=str(file_rel_path),
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=f"Upload file {file_rel_path}"
                    )
                    logger.info(f"Uploaded file: {file_rel_path}")
                except Exception as e2:
                    logger.error(f"Failed to upload file {file_rel_path}: {str(e2)}")
                time.sleep(10)
            elif file.is_dir() and file.name not in SKIP_FOLDERS:
                upload_folder_or_files(api, file, repo_id, base_path)  # Recursively handle subfolders

def deploy_all_top_folders():
    base_path = Path("S:/HCSearchModel/HC-Search-Models/HarmocareModels")
    token = "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"
    repo_id = "Sumedh3456/HC-Search"

    login(token=token, write_permission=True, add_to_git_credential=True)
    api = HfApi()

    for item in base_path.iterdir():
        if item.is_dir() and item.name not in SKIP_FOLDERS:
            files = [f for f in item.rglob("*") if f.is_file() and not f.name.endswith(('.pyc', '.log')) and "__pycache__" not in f.parts]
            if not files:
                logger.info(f"Skipping empty or ignored folder: {item.name}")
                continue
            upload_folder_or_files(api, item, repo_id, base_path)
        elif item.is_file() and item.name not in SKIP_FILES:
            logger.info(f"Uploading file: {item.name}")
            try:
                api.upload_file(
                    path_or_fileobj=str(item),
                    path_in_repo=item.name,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Upload file {item.name}"
                )
                logger.info(f"Uploaded file: {item.name}")
            except Exception as e:
                logger.error(f"Failed to upload file {item.name}: {str(e)}")
            time.sleep(10)

if __name__ == "__main__":
    base_path = Path("S:/HCSearchModel/HC-Search-Models")
    token = "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"
    repo_id = "Sumedh3456/HC-Search"

    login(token=token, write_permission=True, add_to_git_credential=True)
    api = HfApi()

    pgvector_folder = base_path / "PGVector"
    if pgvector_folder.exists() and pgvector_folder.is_dir():
        upload_folder_or_files(api, pgvector_folder, repo_id, base_path)
    else:
        logger.error("PGVector folder does not exist!")
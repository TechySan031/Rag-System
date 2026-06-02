"""
HF Hub Persistence — sync ChromaDB data and uploads to a Hugging Face Dataset repo.

HF Spaces uses ephemeral Docker storage: every restart (deploy, crash, idle timeout)
wipes /app/chroma_data/ and /app/uploads/. This module syncs those directories to a
private HF Dataset repo so documents survive restarts.

Flow:
    Startup  → restore_from_hub()  → download chroma_data/ + uploads/ from dataset repo
    Upload   → sync_to_hub()       → push chroma_data/ + new file to dataset repo

Design decisions:
    - Uses huggingface_hub.HfApi (official SDK) for all operations
    - Graceful degradation: sync failures are logged, never crash the app
    - Auto-creates the dataset repo (private) on first sync
    - Sync is synchronous but fast (~1-3s for small datasets) — acceptable for
      real-time sync after upload since the upload itself takes 3-10s
    - Dev mode: persistence disabled by default (HF_PERSISTENCE_ENABLED=false)
"""
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import (
    HF_PERSISTENCE_ENABLED,
    HF_DATASET_REPO,
    HF_TOKEN,
    CHROMA_DIR,
    UPLOAD_DIR,
)

# --- Sync status tracker ---
_sync_status = {
    "enabled": HF_PERSISTENCE_ENABLED,
    "repo": HF_DATASET_REPO,
    "last_sync_at": None,
    "last_sync_duration_ms": None,
    "last_sync_status": None,  # "success" | "error" | None
    "last_sync_error": None,
    "last_restore_at": None,
    "last_restore_status": None,
    "total_syncs": 0,
    "total_sync_errors": 0,
}


def _get_api():
    """Lazy-import and return HfApi instance."""
    from huggingface_hub import HfApi
    token = HF_TOKEN or None  # None = use default HF_TOKEN env or cached login
    return HfApi(token=token)


def _ensure_repo_exists(api) -> bool:
    """Create the dataset repo if it doesn't exist. Returns True if ready."""
    try:
        api.repo_info(repo_id=HF_DATASET_REPO, repo_type="dataset")
        return True
    except Exception:
        try:
            api.create_repo(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                private=True,
                exist_ok=True,
            )
            print(f"[persistence] Created dataset repo: {HF_DATASET_REPO}")
            return True
        except Exception as e:
            print(f"[persistence] ERROR: Cannot create repo {HF_DATASET_REPO}: {e}")
            return False


def restore_from_hub() -> dict:
    """
    Download persisted data from HF Dataset repo to local directories.
    Called once at startup, BEFORE BM25 index rebuild.

    Downloads:
        - chroma_data/ → CHROMA_DIR (ChromaDB persistent storage)
        - uploads/     → UPLOAD_DIR (original uploaded files)

    Returns:
        dict with status and details
    """
    if not HF_PERSISTENCE_ENABLED:
        print("[persistence] Disabled (HF_PERSISTENCE_ENABLED=false)")
        return {"status": "disabled"}

    print(f"[persistence] Restoring from {HF_DATASET_REPO}...")
    start = time.perf_counter()

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError

        token = HF_TOKEN or None

        try:
            # Download entire dataset repo to a temp location
            local_dir = snapshot_download(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                token=token,
                local_dir=str(CHROMA_DIR.parent / ".hf_restore_tmp"),
            )
        except RepositoryNotFoundError:
            print("[persistence] Dataset repo not found — first deploy, nothing to restore.")
            _sync_status["last_restore_at"] = datetime.now(timezone.utc).isoformat()
            _sync_status["last_restore_status"] = "no_repo"
            return {"status": "no_repo", "message": "First deploy — no data to restore"}
        except Exception as e:
            print(f"[persistence] WARNING: Could not download from hub: {e}")
            _sync_status["last_restore_at"] = datetime.now(timezone.utc).isoformat()
            _sync_status["last_restore_status"] = "error"
            return {"status": "error", "error": str(e)}

        local_path = Path(local_dir)
        restored_files = 0

        # Restore chroma_data/
        src_chroma = local_path / "chroma_data"
        if src_chroma.exists() and any(src_chroma.iterdir()):
            # Clear local chroma_data and copy from snapshot
            if CHROMA_DIR.exists():
                shutil.rmtree(CHROMA_DIR)
            shutil.copytree(src_chroma, CHROMA_DIR)
            restored_files += sum(1 for _ in CHROMA_DIR.rglob("*") if _.is_file())
            print(f"[persistence] Restored chroma_data/ ({restored_files} files)")

        # Restore uploads/
        src_uploads = local_path / "uploads"
        if src_uploads.exists() and any(src_uploads.iterdir()):
            UPLOAD_DIR.mkdir(exist_ok=True)
            upload_count = 0
            for f in src_uploads.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    dest = UPLOAD_DIR / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)
                        upload_count += 1
            restored_files += upload_count
            print(f"[persistence] Restored uploads/ ({upload_count} files)")

        # Clean up temp directory
        tmp_dir = CHROMA_DIR.parent / ".hf_restore_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

        duration_ms = (time.perf_counter() - start) * 1000
        print(f"[persistence] Restore complete: {restored_files} files in {duration_ms:.0f}ms")

        _sync_status["last_restore_at"] = datetime.now(timezone.utc).isoformat()
        _sync_status["last_restore_status"] = "success"

        return {
            "status": "success",
            "restored_files": restored_files,
            "duration_ms": round(duration_ms, 2),
        }

    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        print(f"[persistence] ERROR during restore: {e}")
        _sync_status["last_restore_at"] = datetime.now(timezone.utc).isoformat()
        _sync_status["last_restore_status"] = "error"
        return {"status": "error", "error": str(e), "duration_ms": round(duration_ms, 2)}


def sync_to_hub(uploaded_filepath: str | None = None) -> dict:
    """
    Push current ChromaDB data and uploads to HF Dataset repo.
    Called after each successful document upload.

    Args:
        uploaded_filepath: Path to the newly uploaded file (optional, for logging)

    Returns:
        dict with sync status and details
    """
    if not HF_PERSISTENCE_ENABLED:
        return {"status": "disabled"}

    start = time.perf_counter()
    _sync_status["total_syncs"] += 1

    try:
        api = _get_api()

        if not _ensure_repo_exists(api):
            raise RuntimeError(f"Cannot access or create repo: {HF_DATASET_REPO}")

        # Upload chroma_data/ directory
        if CHROMA_DIR.exists() and any(CHROMA_DIR.rglob("*")):
            api.upload_folder(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                folder_path=str(CHROMA_DIR),
                path_in_repo="chroma_data",
                commit_message=f"Sync chroma_data after upload: {Path(uploaded_filepath).name if uploaded_filepath else 'manual'}",
            )

        # Upload uploads/ directory (original files for reference)
        if UPLOAD_DIR.exists() and any(UPLOAD_DIR.iterdir()):
            api.upload_folder(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                folder_path=str(UPLOAD_DIR),
                path_in_repo="uploads",
                commit_message=f"Sync uploads: {Path(uploaded_filepath).name if uploaded_filepath else 'manual'}",
            )

        duration_ms = (time.perf_counter() - start) * 1000

        _sync_status["last_sync_at"] = datetime.now(timezone.utc).isoformat()
        _sync_status["last_sync_duration_ms"] = round(duration_ms, 2)
        _sync_status["last_sync_status"] = "success"
        _sync_status["last_sync_error"] = None

        filename = Path(uploaded_filepath).name if uploaded_filepath else "manual"
        print(f"[persistence] Synced to {HF_DATASET_REPO} in {duration_ms:.0f}ms (trigger: {filename})")

        return {
            "status": "success",
            "repo": HF_DATASET_REPO,
            "duration_ms": round(duration_ms, 2),
        }

    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        _sync_status["total_sync_errors"] += 1
        _sync_status["last_sync_at"] = datetime.now(timezone.utc).isoformat()
        _sync_status["last_sync_duration_ms"] = round(duration_ms, 2)
        _sync_status["last_sync_status"] = "error"
        _sync_status["last_sync_error"] = str(e)[:200]

        print(f"[persistence] WARNING: Sync failed ({duration_ms:.0f}ms): {e}")
        return {
            "status": "error",
            "error": str(e)[:200],
            "duration_ms": round(duration_ms, 2),
        }


def get_sync_status() -> dict:
    """Return current sync status for health endpoint."""
    return {**_sync_status}

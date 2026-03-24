import os
import sys
import tarfile
import shutil
import urllib.request

DOWNLOAD_URL = "https://services.healthtech.dtu.dk/services/NetSolP-1.0/netsolp-1.0.ALL.tar.gz"
ARCHIVE_NAME = "netsolp-1.0.ALL.tar.gz"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

EXPECTED_EXTENSIONS = (".onnx", ".pkl")


def download_with_progress(url, dest):
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct}%)")
            sys.stdout.flush()

    print(f"Downloading from {url}")
    urllib.request.urlretrieve(url, dest, reporthook)
    print()


def extract_models(archive_path, dest_dir):
    print(f"Extracting model files to {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if any(member.name.endswith(ext) for ext in EXPECTED_EXTENSIONS):
                basename = os.path.basename(member.name)
                member.name = basename
                tar.extract(member, dest_dir)
                print(f"  Extracted: {basename}")


def models_exist():
    if not os.path.isdir(MODELS_DIR):
        return False
    onnx_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".onnx")]
    return len(onnx_files) > 0


if __name__ == "__main__":
    if models_exist():
        print(f"Models already exist in {MODELS_DIR}")
        sys.exit(0)

    download_with_progress(DOWNLOAD_URL, ARCHIVE_NAME)
    extract_models(ARCHIVE_NAME, MODELS_DIR)

    print(f"Cleaning up {ARCHIVE_NAME}")
    os.remove(ARCHIVE_NAME)

    print("Done. Model files:")
    for f in sorted(os.listdir(MODELS_DIR)):
        print(f"  {f}")

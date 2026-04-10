"""
Script untuk mempersiapkan file yang perlu di-upload ke Colab.

Jalankan dari root project:
    python src/finetuned/prepare_for_colab.py

Output:
    colab_upload/
    ├── src_finetuned.zip    ← source code (upload ke Colab)
    └── README_upload.txt    ← instruksi upload
"""

import zipfile
import shutil
from pathlib import Path


def create_colab_package():
    """Buat zip package untuk di-upload ke Colab."""
    
    root = Path(__file__).parent.parent.parent  # D:\2-Project\AQG
    output_dir = root / "colab_upload"
    output_dir.mkdir(exist_ok=True)
    
    # 1. Zip src/finetuned/
    print("Creating src_finetuned.zip...")
    zip_path = output_dir / "src_finetuned.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        src_dir = root / "src" / "finetuned"
        for file_path in src_dir.rglob("*"):
            if file_path.is_file() and "__pycache__" not in str(file_path):
                arcname = file_path.relative_to(root)
                zf.write(file_path, arcname)
    
    print(f"✓ {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
    
    # 2. Buat README instruksi
    readme_path = output_dir / "README_upload.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("""INSTRUKSI UPLOAD KE COLAB
========================

File yang perlu di-upload ke Colab:

1. src_finetuned.zip
   → Upload via cell "Upload Source Code" di notebook
   → Akan di-extract ke /content/src/finetuned/

2. Dataset files (upload terpisah):
   Dari: dataset_aqg/output_domain/
   → train.jsonl
   → validation.jsonl
   → test.jsonl

   Dari: dataset_aqg/dataset-task-spesifc/
   → train.jsonl
   → validation.jsonl
   → test.jsonl

URUTAN NOTEBOOK:
1. 01_setup_and_validation.ipynb
2. 02_domain_adaptation.ipynb
3. 03_task_specific_training.ipynb
""")
    
    print(f"✓ {readme_path}")
    print(f"\nFile siap di-upload: {output_dir}")
    print("\nLangkah selanjutnya:")
    print("1. Buka 01_setup_and_validation.ipynb di VS Code")
    print("2. Pastikan kernel = Colab")
    print("3. Jalankan cell install dependencies")
    print("4. Upload src_finetuned.zip dan dataset files")


if __name__ == "__main__":
    create_colab_package()

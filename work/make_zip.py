import os
import zipfile
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(BASE_DIR, "student_resource")
OUTPUT_DIR = os.path.join(RES_DIR, "output")
CODE_DIR = os.path.join(RES_DIR, "code", "business_entity_resolution")

zip_path = os.path.join(BASE_DIR, "submission.zip")
print(f"Creating {zip_path}...")
t0 = time.time()

files_to_pack = [
    (os.path.join(OUTPUT_DIR, "matching_results.tsv"), "output/matching_results.tsv"),
    (os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), "output/candidate_pairs.tsv"),
    (os.path.join(CODE_DIR, "README.md"), "code/business_entity_resolution/README.md"),
    (os.path.join(CODE_DIR, "requirements.txt"), "code/business_entity_resolution/requirements.txt"),
    (os.path.join(CODE_DIR, "src", "features.py"), "code/business_entity_resolution/src/features.py"),
    (os.path.join(CODE_DIR, "src", "inference.py"), "code/business_entity_resolution/src/inference.py"),
    (os.path.join(CODE_DIR, "src", "train.py"), "code/business_entity_resolution/src/train.py"),
    (os.path.join(CODE_DIR, "src", "metric.py"), "code/business_entity_resolution/src/metric.py"),
    (os.path.join(CODE_DIR, "models", "lightgbm_model.txt"), "code/business_entity_resolution/models/lightgbm_model.txt"),
    (os.path.join(RES_DIR, "Documentation_template.md"), "Documentation_template.md"),
]

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for src, arcname in files_to_pack:
        if os.path.exists(src):
            sz = os.path.getsize(src)
            print(f"  Adding {arcname} ({sz:,d} bytes)...", flush=True)
            zipf.write(src, arcname)
        else:
            print(f"  WARNING: {src} missing!", flush=True)

sz_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"SUCCESS! Created {zip_path} ({sz_mb:.1f} MB) in {time.time()-t0:.1f}s.", flush=True)

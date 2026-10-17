import os
import sys
import shutil
import subprocess
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "student_resource", "output")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TEST_DIR = os.path.join(BASE_DIR, "student_resource", "dataset", "test")

print("Checking results/candidate_pairs.tsv...")
res_cand = os.path.join(RESULTS_DIR, "candidate_pairs.tsv")
if os.path.exists(res_cand):
    with open(res_cand, "rb") as fp:
        lines = sum(1 for _ in fp)
    print(f"results/candidate_pairs.tsv has {lines:,} lines ({os.path.getsize(res_cand):,d} bytes).")
    
    # 1.732,544 S1 + 1 header = 1,732,545 lines
    if lines == 1732545:
        print("results/candidate_pairs.tsv is 100% complete! Copying to student_resource/output...")
        dest_cand = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
        shutil.copyfile(res_cand, dest_cand)
        print(f"Copied to {dest_cand} ({os.path.getsize(dest_cand):,d} bytes).")

res_match = os.path.join(RESULTS_DIR, "matching_results.tsv")
if os.path.exists(res_match):
    with open(res_match, "rb") as fp:
        m_lines = sum(1 for _ in fp)
    print(f"results/matching_results.tsv has {m_lines:,} lines ({os.path.getsize(res_match):,d} bytes).")
    dest_match = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    if not os.path.exists(dest_match) or os.path.getsize(dest_match) != os.path.getsize(res_match):
        shutil.copyfile(res_match, dest_match)
        print(f"Copied matching_results.tsv to {dest_match}.")

print("\nRunning official validator...")
validator_script = os.path.join(BASE_DIR, "student_resource", "utils", "validate_submission.py")
matching_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
candidate_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

cmd = [
    sys.executable, validator_script,
    "--matching", matching_file,
    "--candidate", candidate_file,
    "--test-dir", TEST_DIR
]
print(f"Executing: {' '.join(cmd)}")
res = subprocess.run(cmd, capture_output=True, text=True)
print("\n=== VALIDATOR OUTPUT ===")
print(res.stdout)
if res.stderr:
    print("Stderr:", res.stderr)

if res.returncode == 0:
    print("\n*** VALIDATION STATUS: PASS! ***\n")
    print("Creating submission.zip...")
    zip_path = os.path.join(BASE_DIR, "submission.zip")
    files_to_pack = [
        (matching_file, "output/matching_results.tsv"),
        (candidate_file, "output/candidate_pairs.tsv"),
        (os.path.join(BASE_DIR, "student_resource", "code", "business_entity_resolution", "README.md"), "code/business_entity_resolution/README.md"),
        (os.path.join(BASE_DIR, "student_resource", "code", "business_entity_resolution", "requirements.txt"), "code/business_entity_resolution/requirements.txt"),
        (os.path.join(BASE_DIR, "student_resource", "code", "business_entity_resolution", "src", "features.py"), "code/business_entity_resolution/src/features.py"),
        (os.path.join(BASE_DIR, "student_resource", "code", "business_entity_resolution", "src", "inference.py"), "code/business_entity_resolution/src/inference.py"),
        (os.path.join(BASE_DIR, "student_resource", "code", "business_entity_resolution", "src", "train.py"), "code/business_entity_resolution/src/train.py"),
        (os.path.join(BASE_DIR, "student_resource", "code", "business_entity_resolution", "src", "metric.py"), "code/business_entity_resolution/src/metric.py"),
        (os.path.join(BASE_DIR, "student_resource", "Documentation_template.md"), "Documentation_template.md"),
    ]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for src, arcname in files_to_pack:
            if os.path.exists(src):
                sz = os.path.getsize(src)
                print(f"  Adding {arcname} ({sz:,d} bytes)...")
                zipf.write(src, arcname)
            else:
                print(f"  WARNING: {src} missing!")
    print(f"\nSUCCESS! Submission package created at {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.1f} MB)!")
else:
    print(f"Validation failed with code {res.returncode}")

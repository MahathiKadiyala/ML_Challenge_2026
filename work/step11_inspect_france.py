"""
Inspect French test records to verify blocking keys, normalization, and candidate generation.
"""
import os
import polars as pl
import re
import unidecode

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")

def run():
    print("Loading French test records...")
    s1_fr = pl.read_csv(os.path.join(DATA_DIR, "test", "test_source1.tsv"), separator="\t").filter(pl.col("country") == "France")
    s2_fr = pl.read_csv(os.path.join(DATA_DIR, "test", "test_source2.tsv"), separator="\t").filter(pl.col("country") == "France")
    s3_fr = pl.read_csv(os.path.join(DATA_DIR, "test", "test_source3.tsv"), separator="\t").filter(pl.col("country") == "France")
    
    print(f"France S1: {len(s1_fr):,}, S2: {len(s2_fr):,}, S3: {len(s3_fr):,}")
    
    print("\nSample French S1 records:")
    for row in s1_fr.head(5).iter_rows():
        print(f"ID: {row[0]} | Name: '{row[1]}' | Addr: '{row[2]}'")
        
    print("\nSample French S2 records:")
    for row in s2_fr.head(5).iter_rows():
        print(f"ID: {row[0]} | Name: '{row[1]}' | Addr: '{row[2]}'")
        
    print("\nSample French S3 records:")
    for row in s3_fr.head(5).iter_rows():
        print(f"ID: {row[0]} | Name: '{row[1]}' | Addr: '{row[2]}'")

if __name__ == "__main__":
    run()

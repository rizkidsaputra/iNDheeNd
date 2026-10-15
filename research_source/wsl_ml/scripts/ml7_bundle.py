#!/usr/bin/env python3
"""Create a new ML-7 release around an unchanged copy of the ML-6 distribution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args()
    repo=args.repo.resolve(); dest=args.destination.resolve()
    dest.mkdir(parents=True,exist_ok=False)
    for directory,patterns in {"src":["ml7*"],"scripts":["ml7*.py","poc11c_apps_ml7_v8.cpp"],
                               "tests":["test_ml7_online.py"],"docs":["ml7_contract.md"]}.items():
        (dest/directory).mkdir()
        for pattern in patterns:
            for p in (repo/directory).glob(pattern):
                if p.is_file(): shutil.copy2(p,dest/directory/p.name)
    package=repo/"dist/ml6_ndn_ifa_offline"
    vendor=dest/"vendor/ml6_ndn_ifa_offline"
    shutil.copytree(package,vendor,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for p in vendor.rglob("*"):
        if p.is_file() and p.read_bytes()!=(package/p.relative_to(vendor)).read_bytes():
            raise ValueError("ML-6 copy mismatch")
    shutil.copytree(repo/"results/ml7",dest/"evidence",ignore=shutil.ignore_patterns("__pycache__"))
    registry_hash=hashlib.sha256((vendor/"registry.json").read_bytes()).hexdigest()
    (dest/"README.md").write_text("# ML-7 Collector M runtime\n\nSee docs/ml7_contract.md and evidence/ for scope and acceptance.\n\nML-6 vendor files are unchanged. This release is experimental.\n\nPinned ML-6 registry SHA-256: "+registry_hash+"\n")
    hashes={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(dest.rglob("*")) if p.is_file()}
    (dest/"SHA256.json").write_text(json.dumps(hashes,indent=2))
    archive=dest.parent/(dest.name+".tar.gz")
    with tarfile.open(archive,"x:gz") as tar: tar.add(dest,arcname=dest.name)
    with tarfile.open(archive,"r:gz") as tar:
        for relative,digest in hashes.items():
            content=tar.extractfile(dest.name+"/"+relative).read()
            if hashlib.sha256(content).hexdigest()!=digest: raise ValueError("Archive checksum mismatch")
    print(json.dumps({"archive":str(archive),"sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
                      "files_verified":len(hashes),"registry_sha256":registry_hash}))


if __name__=="__main__": main()

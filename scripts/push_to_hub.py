"""`make publish-hf`: upload adapter + model card + schema to Hugging Face, after you type "yes".

Defaults to a PRIVATE repo; pass --public to make it public. Reads HF_TOKEN from .env and never
prints it. Nothing is uploaded unless you confirm.
"""

from __future__ import annotations

import argparse
import json
import sys

from joblens.config import ARTIFACTS, DOCS, ROOT, hf_token
from joblens.release import HF_REPO, stage_release


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--adapter", default="")
    args = ap.parse_args()

    best_file = ARTIFACTS / "training" / "best.json"
    best = json.loads(best_file.read_text())["best"] if best_file.exists() else "r16"
    adapter = ROOT / args.adapter if args.adapter else ARTIFACTS / "adapters" / best
    out = ARTIFACTS / "release"
    files = stage_release(
        adapter, DOCS / "MODEL_CARD.md", ROOT / "schema" / "job_posting.schema.json", out
    )
    print(f"Staged in {out}:")
    for f in files:
        print("  ", f)
    vis = "PUBLIC" if args.public else "private"
    print(f"\nAbout to upload these files to https://huggingface.co/{args.repo} ({vis}).")
    if input('Type "yes" to upload, anything else to cancel: ').strip() != "yes":
        print("Cancelled. Nothing was uploaded.")
        return 1
    token = hf_token()
    if not token:
        print("HF_TOKEN missing in .env")
        return 1
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, private=not args.public, exist_ok=True, repo_type="model")
    info = api.upload_folder(
        repo_id=args.repo,
        folder_path=str(out),
        commit_message="JobLens adapter v1.0 (schema 1.0.0)",
        repo_type="model",
    )
    print("Uploaded:", getattr(info, "commit_url", info))
    return 0


if __name__ == "__main__":
    sys.exit(main())

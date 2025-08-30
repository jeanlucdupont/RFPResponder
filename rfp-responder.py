#!/usr/bin/env python3
"""
rfp-responder.py

Reads an Excel questionnaire from S3 with columns:
  question | answer | comment

For each 'question', finds the most similar canonical_question in
answers_with_candidates.yaml using Amazon Bedrock Titan Text Embeddings v2,
retrieves its 'answer', and sets the 'comment' to a summary of the top policy
candidate: source_doc, section, text_preview, nda_tier, score.

Usage:
  python rfp-responder.py `
    --xlsx s3://my-bucket/path/questionnaire.xlsx `
    --yaml s3://my-bucket/path/answers_with_candidates.yaml `
    --out  s3://my-bucket/path/questionnaire_filled.xlsx `
    --region us-east-1 --profile XXXXX `
    --min-sim 0.78

Requirements:
  - boto3, botocore, pandas, PyYAML, numpy, openpyxl
"""

import argparse
import io
import json
import sys
import re
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import boto3
except Exception:
    boto3 = None


# ------------------------- S3 helpers -------------------------
def is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.lower().startswith("s3://")


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    assert uri.lower().startswith("s3://")
    no_scheme = uri[5:]
    bucket, key = no_scheme.split("/", 1)
    return bucket, key


def s3_client(profile=None, region=None):
    if boto3 is None:
        raise RuntimeError("boto3 is required for S3 operations. pip install boto3")
    sess = boto3.Session(profile_name=profile, region_name=region)
    return sess.client("s3")


def read_bytes_any(path_or_uri: str, profile=None, region=None) -> bytes:
    if is_s3_uri(path_or_uri):
        s3 = s3_client(profile, region)
        b, k = parse_s3_uri(path_or_uri)
        obj = s3.get_object(Bucket=b, Key=k)
        return obj["Body"].read()
    with open(path_or_uri, "rb") as f:
        return f.read()


def read_text_any(path_or_uri: str, profile=None, region=None) -> str:
    data = read_bytes_any(path_or_uri, profile, region)
    return data.decode("utf-8")


def write_bytes_any(path_or_uri: str, data: bytes, profile=None, region=None, content_type=None):
    if is_s3_uri(path_or_uri):
        s3 = s3_client(profile, region)
        b, k = parse_s3_uri(path_or_uri)
        kwargs = {"Bucket": b, "Key": k, "Body": data}
        if content_type:
            kwargs["ContentType"] = content_type
        s3.put_object(**kwargs)
        return
    with open(path_or_uri, "wb") as f:
        f.write(data)


# ------------------------- Bedrock Embeddings -------------------------
def bedrock_client(region, profile=None):
    if boto3 is None:
        raise RuntimeError("boto3 is required for Bedrock. pip install boto3")
    if not region:
        region = "us-east-1"
    sess = boto3.Session(profile_name=profile, region_name=region)
    return sess.client("bedrock-runtime")


def embed_texts(bedrock, texts, model_id="amazon.titan-embed-text-v2:0") -> np.ndarray:
    """
    Titan v2 embedding: one string per call, returns 'embedding' list[float].
    """
    vecs = []
    for t in texts:
        body = {"inputText": t}
        resp = bedrock.invoke_model(modelId=model_id, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        emb = payload.get("embedding") or payload.get("embeddingsByType", {}).get("float")
        if not emb:
            raise RuntimeError(f"No embedding returned for: {t[:80]}... (keys={list(payload.keys())})")
        vecs.append(emb)
    return np.array(vecs, dtype=np.float32)


# ------------------------- YAML loader -------------------------
def load_yaml_answers(path_or_uri: str, profile=None, region=None) -> List[Dict[str, Any]]:
    """
    Expects a list of records, each like:
      - canonical_id: ...
        canonical_question: ...
        topics: ...
        answer: ...
        candidates: [ {source_doc, section, text_preview, nda_tier, score, ...}, ... ]
    """
    txt = read_text_any(path_or_uri, profile, region)
    data = yaml.safe_load(txt)
    if not isinstance(data, list):
        raise ValueError("answers_with_candidates.yaml must be a list of records")
    return data


def normalize_space(s) -> str:
    """
    Coerce to string and collapse whitespace.
    Handles None, int, float, bool safely.
    """
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip())


# ------------------------- Matching -------------------------
def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Returns cosine similarity matrix between rows of A (n x d) and B (m x d).
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return np.matmul(A_norm, B_norm.T)


def pick_comment_from_record(rec: Dict[str, Any]) -> str:
    """
    Build the comment string from the top candidate in rec['candidates'], if any.
    Format includes: source_doc, section, text_preview, nda_tier, score
    """
    cands = rec.get("candidates") or []
    if not cands:
        return ""
    top = cands[0]
    source_doc = top.get("source_doc") or ""
    section = top.get("section") or ""
    text_preview = (top.get("text_preview") or "").replace("\n", " ")
    nda_tier = top.get("nda_tier") or ""
    score = top.get("score")
    score_str = f"{float(score):.4f}" if score is not None else ""
    # Keep it compact for Excel cells
    return f"source_doc={source_doc} | section={section} | nda_tier={nda_tier} | score={score_str} | text_preview={text_preview}"


# ------------------------- Main flow -------------------------
def main():
    ap = argparse.ArgumentParser(description="Fill questionnaire Excel from YAML answers (via Titan embeddings).")
    ap.add_argument("--xlsx", required=True, help="Input questionnaire Excel path (S3 or local). Requires columns: question, answer, comment")
    ap.add_argument("--yaml", required=True, help="answers_with_candidates.yaml path (S3 or local)")
    ap.add_argument("--out", required=True, help="Output Excel path (S3 or local)")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--model-id", default="amazon.titan-embed-text-v2:0")
    ap.add_argument("--min-sim", type=float, default=0.78, help="Minimum cosine similarity to accept a match")
    args = ap.parse_args()

    # Load YAML records
    records = load_yaml_answers(args.yaml, profile=args.profile, region=args.region)

    # Extract canonical questions and answers
    canon_questions = [normalize_space(r.get("canonical_question") or "") for r in records]
    canon_answers = [normalize_space(r.get("answer") or "") for r in records]

    # Validate
    if not canon_questions or all(q == "" for q in canon_questions):
        print("No canonical questions found in YAML.", file=sys.stderr)
        sys.exit(2)

    # Read questionnaire Excel
    xlsx_bytes = read_bytes_any(args.xlsx, profile=args.profile, region=args.region)
    df = pd.read_excel(io.BytesIO(xlsx_bytes), engine="openpyxl")

    # Ensure required columns exist
    for col in ["question", "answer", "comment"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in the Excel file")

    # Prepare embeddings
    bed = bedrock_client(args.region, profile=args.profile)
    canon_vecs = embed_texts(bed, canon_questions, model_id=args.model_id)

    # Process each questionnaire row
    q_texts = [normalize_space(str(q)) for q in df["question"].astype(str).tolist()]
    q_vecs = embed_texts(bed, q_texts, model_id=args.model_id)

    # Compute similarity (rows: questionnaire; cols: canonical)
    sims = cosine_sim_matrix(q_vecs, canon_vecs)  # shape (n_q, n_c)

    best_idxs = sims.argmax(axis=1)
    best_scores = sims[np.arange(sims.shape[0]), best_idxs]

    filled_answers = []
    filled_comments = []

    for i, (idx, s) in enumerate(zip(best_idxs, best_scores)):
        if float(s) < args.min_sim:
            # No confident match
            filled_answers.append(df.at[i, "answer"])   # or "" if you want blank
            filled_comments.append("NO MATCH")
        else:
            rec = records[idx]
            ans = rec.get("answer") or ""
            comment = pick_comment_from_record(rec)
            filled_answers.append(ans)
            filled_comments.append(comment)

    df["answer"] = filled_answers
    df["comment"] = filled_comments

    # Write Excel out
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Questionnaire")
    bio.seek(0)

    write_bytes_any(args.out, bio.read(), profile=args.profile, region=args.region,
                    content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    print(f"Completed. Wrote: {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
genlib.py

- Loads canonical answers (answers.yaml) and policy chunks (policy-chunks.jsonl) from local or S3
- Uses Amazon Bedrock Titan Text Embeddings v2 to embed (one string per call; no embeddingConfig)
- Ranks chunk -> canonical similarity and writes a SINGLE YAML output with the final structure:
  [
    {
      canonical_id: str,
      canonical_question: str,
      topics: <from answers.yaml, optional>,
      answer: <from answers.yaml, optional>,
      candidates: [
        {
          id: str,
          source_doc: str,
          section: str,
          page_range: str|null,
          nda_tier: str,
          score: float,
          text_preview: str
        }, ...
      ]
    }, ...
  ]

Usage:
python genlib.py `
  --answers s3://<bucket>/library/answers.yaml `
  --chunks  s3://<bucket>/policy-chunks/policy-chunks.jsonl `
  --out     s3://<bucket>/library/answers_with_candidates.yaml `
  --min-sim 0.84 --topk 3 `
  --region us-east-1 --profile XXXXXX
"""
import argparse, json, sys, yaml, hashlib
from typing import List, Dict, Any
import numpy as np

try:
    import boto3
except Exception:
    boto3 = None

# ------------------------- S3 helpers -------------------------
def is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.lower().startswith("s3://")

def parse_s3_uri(uri: str):
    assert uri.lower().startswith("s3://")
    no_scheme = uri[5:]
    bucket, key = no_scheme.split("/", 1)
    return bucket, key

def s3_client(profile=None, region=None):
    if boto3 is None:
        raise RuntimeError("boto3 is required for S3 operations. pip install boto3")
    sess = boto3.Session(profile_name=profile, region_name=region)
    return sess.client("s3")

def read_text_any(path_or_uri: str, profile=None, region=None) -> str:
    if is_s3_uri(path_or_uri):
        s3 = s3_client(profile, region)
        b, k = parse_s3_uri(path_or_uri)
        obj = s3.get_object(Bucket=b, Key=k)
        return obj["Body"].read().decode("utf-8")
    with open(path_or_uri, "r", encoding="utf-8") as f:
        return f.read()

def write_text_any(path_or_uri: str, text: str, profile=None, region=None, content_type="text/yaml"):
    if is_s3_uri(path_or_uri):
        s3 = s3_client(profile, region)
        b, k = parse_s3_uri(path_or_uri)
        s3.put_object(Bucket=b, Key=k, Body=text.encode("utf-8"), ContentType=content_type)
        return
    with open(path_or_uri, "w", encoding="utf-8") as f:
        f.write(text)

# ------------------------- Bedrock -------------------------
def bedrock_client(region, profile=None):
    if boto3 is None:
        raise RuntimeError("boto3 is required for Bedrock. pip install boto3")
    if not region:
        region = "us-east-1"
    sess = boto3.Session(profile_name=profile, region_name=region)
    return sess.client("bedrock-runtime")

def embed_texts(bedrock, texts, model_id="amazon.titan-embed-text-v2:0"):
    """
    Titan v2 request body uses only {"inputText": "..."} and returns a single 1024-dim vector
    under 'embedding' (sometimes mirrored under embeddingsByType.float).
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

# ------------------------- Data parsing -------------------------
def load_answers(path_or_uri: str, profile=None, region=None) -> List[Dict[str, Any]]:
    txt = read_text_any(path_or_uri, profile, region)
    data = yaml.safe_load(txt)
    if not isinstance(data, list):
        raise ValueError("answers.yaml must be a list of records")
    return data

def ensure_chunk_id(c: Dict[str, Any]) -> str:
    if "id" in c and c["id"]:
        return c["id"]
    base = f"{c.get('source_doc','')}/{c.get('section','')}/{(c.get('text','')[:80])}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    cid = f"{c.get('source_doc','doc')}:{c.get('section','sec')}:{h}"
    c["id"] = cid
    return cid

def load_chunks(path_or_uri: str, profile=None, region=None) -> List[Dict[str, Any]]:
    txt = read_text_any(path_or_uri, profile, region)
    chunks = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        ensure_chunk_id(c)
        c["nda_tier"] = (c.get("nda_tier") or "under-nda").lower()
        chunks.append(c)
    return chunks

# ------------------------- Scoring -------------------------
def keywords_from_text(t: str):
    base = [
        "encrypt","at rest","in transit","kms","tls","https",
        "mfa","privileged","role","least privilege","access",
        "log","monitor","audit","siem","retention",
        "backup","recovery","snapshot","rto","rpo","bcp",
        "incident","forensic","breach","vendor","third party",
        "sdlc","code review","appsec","firewall","segmentation"
    ]
    low = t.lower()
    return [k for k in base if k in low]

def rerank(candidates, query_text: str, controls_wanted=None):
    if controls_wanted is None:
        controls_wanted = set()
    kw = set(keywords_from_text(query_text))
    reranked = []
    for base_score, s in candidates:
        text = (s.get("text") or "").lower()
        ctrl_score = 0.0
        if s.get("controls"):
            inter = controls_wanted.intersection(set(s["controls"]))
            ctrl_score = 0.5 * len(inter)
        kw_score = 0.0
        for k in kw:
            if k in text: kw_score += 0.1
        final = base_score + ctrl_score + kw_score
        reranked.append((final, s))
    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked

# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser(description="AI-assisted policy snippet linking (single YAML out)")
    ap.add_argument("--answers", required=True, help="answers.yaml (list of canonical Q/As)")
    ap.add_argument("--chunks", required=True, help="policy-chunks.jsonl (one JSON per line)")
    ap.add_argument("--out", required=True, help="Unified YAML with answers + ranked candidates")
    ap.add_argument("--process-all", action="store_true", help="Process all answers even if already linked")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--min-sim", type=float, default=0.84)
    ap.add_argument("--allow-nda", default="public,under-nda")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--model-id", default="amazon.titan-embed-text-v2:0")
    args = ap.parse_args()

    allow_nda = set([s.strip().lower() for s in args.allow_nda.split(",") if s.strip()])

    # Load inputs
    answers = load_answers(args.answers, profile=args.profile, region=args.region)
    chunks  = load_chunks(args.chunks,  profile=args.profile, region=args.region)
    chunks = [c for c in chunks if c.get("nda_tier","under-nda").lower() in allow_nda]

    if not chunks:
        write_text_any(args.out, yaml.safe_dump([], sort_keys=False, allow_unicode=True), profile=args.profile, region=args.region)
        print("No chunks after NDA filter."); sys.exit(3)

    # Build query list
    queries, idx_map = [], []
    for i, rec in enumerate(answers):
        # Respect existing links unless process-all is set
        if not args.process_all and (rec.get("default", {}) or {}).get("policy_snippet_id"):
            continue
        q = (rec.get("canonical_question") or "").strip()
        short = (rec.get("default", {}).get("short_comment") or "").strip()
        qtext = (q + " " + short).strip()
        if not qtext: 
            continue
        queries.append(qtext); idx_map.append(i)

    if not queries:
        # Still produce a file that mirrors input but with empty candidates (nothing to process)
        unified = []
        for rec in answers:
            unified.append({
                "canonical_id": rec.get("id") or rec.get("canonical_id"),
                "canonical_question": rec.get("canonical_question"),
                "topics": rec.get("topics"),
                "answer": rec.get("answer_type") or (rec.get("default", {}) or {}).get("answer_type"),
                "candidates": []
            })
        write_text_any(args.out, yaml.safe_dump(unified, sort_keys=False, allow_unicode=True), profile=args.profile, region=args.region)
        print("Nothing to process; wrote passthrough structure.")
        return

    # Embeddings
    bed = bedrock_client(args.region, profile=args.profile)
    q_vecs = embed_texts(bed, queries, model_id=args.model_id)
    c_texts = [c.get("text","") for c in chunks]
    c_vecs = embed_texts(bed, c_texts, model_id=args.model_id)

    norms = np.linalg.norm(c_vecs, axis=1) + 1e-8

    # Prepare a copy of answers to decorate with candidates
    unified = []
    for i, rec in enumerate(answers):
        print(f"rec: {rec}")
        unified.append({
            "canonical_id": rec.get("id") or rec.get("canonical_id"),
            "canonical_question": rec.get("canonical_question"),
            "topics": rec.get("topics"),
            "answer": rec.get("answer_type") or (rec.get("default", {}) or {}).get("answer_type"),
            "candidates": []  # to be filled if this rec was processed
        })

    # Scoring + topk selection
    for qi, qvec in enumerate(q_vecs):
        rec = answers[idx_map[qi]]
        qtext = queries[qi]
        sims = np.dot(c_vecs, qvec) / (norms * (np.linalg.norm(qvec)+1e-8))

        # base top10 by cosine, then rerank with simple keyword/control nudges
        top_idx = sims.argsort()[-10:][::-1]
        candidates = [(float(sims[j]), chunks[j]) for j in top_idx]
        candidates = rerank(candidates, qtext)
        top = [(score, c) for score, c in candidates if score >= args.min_sim][:args.topk]

        out_cands = []
        for score, c in top:
            preview = (c.get("text","").strip().replace("\n"," ")[:300] + ("..." if len(c.get("text",""))>300 else ""))
            out_cands.append({
                "id": c["id"],
                "source_doc": c.get("source_doc"),
                "section": c.get("section"),
                "page_range": c.get("page_range"),
                "nda_tier": c.get("nda_tier"),
                "score": round(float(score), 4),
                "text_preview": preview
            })

        # Inject into the unified record at the right index
        unified_idx = idx_map[qi]
        unified[unified_idx]["candidates"] = out_cands

    # Write single YAML
    out_yaml = yaml.safe_dump(unified, sort_keys=False, allow_unicode=True)
    write_text_any(args.out, out_yaml, profile=args.profile, region=args.region)
    print(f"Wrote unified YAML: {args.out}")

if __name__ == "__main__":
    main()

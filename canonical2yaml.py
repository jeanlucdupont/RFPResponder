#!/usr/bin/env python3
'''
xlsx_to_answers.py
Convert an Excel sheet of canonical answers (stored locally or in S3) into a YAML library..
- Supports --in and --out as either local file paths or s3://bucket/key URIs
- Detects Yes/No/Number answers
- Generates stable IDs (from "Ref" when available)
- Copies comments
- (Optional) auto-tags topics based on keywords
Usage examples:
# Local -> S3
python xlsx_to_answers.py --in canonical-answers.xlsx --out s3://my-bucket/library/answers.yaml --topics --approve
# S3 -> S3 (Windows PowerShell)
python xlsx_to_answers.py --in s3://my-bucket/inbox/canonical-answers.xlsx --out s3://my-bucket/library/answers.yaml --topics --approve --profile XXXXXX
# S3 -> Local
python xlsx_to_answers.py --in s3://my-bucket/inbox/canonical-answers.xlsx --out answers.yaml --sheet "sheet1"
'''

import argparse, re, sys, io, os, yaml, pandas as pd
from datetime import datetime

try:
    import boto3
except Exception as e:
    boto3 = None

TOPIC_RULES = [
    (r'encrypt|kms|at rest|in transit|tls|https', "Encryption"),
    (r'iam|identity|mfa|privilege|role|access', "Identity"),
    (r'log|siem|monitor|audit|event', "Logging"),
    (r'vuln|cve|patch|scanner|remediation', "Vulnerability Management"),
    (r'backup|restore|snapshot|recovery', "Backup"),
    (r'disaster|continuity|rto|rpo|bcp', "DR/BCP"),
    (r'incident|ir plan|breach|forensic|playbook', "Incident Response"),
    (r'vendor|third[- ]party|supplier', "Third Party"),
    (r'retention|disposal|erase|purge', "Data Management"),
    (r'privacy|pii|gdpr|ccpa', "Privacy"),
    (r'sdlc|secure development|code review|appsec|owasp', "SDLC"),
    (r'network|firewall|waf|segmentation', "Network Security"),
]

def is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.lower().startswith("s3://")

def parse_s3_uri(uri: str):
    # s3://bucket/key...
    assert uri.lower().startswith("s3://")
    no_scheme = uri[5:]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key

def s3_client(profile=None, region=None):
    if boto3 is None:
        raise RuntimeError("boto3 is required for S3 operations. pip install boto3")
    if profile:
        session = boto3.Session(profile_name=profile, region_name=region)
    else:
        session = boto3.Session(region_name=region)
    return session.client("s3")

def read_excel_any(path_or_uri, sheet_name=None, profile=None, region=None):
    if is_s3_uri(path_or_uri):
        bucket, key = parse_s3_uri(path_or_uri)
        s3 = s3_client(profile, region)
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        bio = io.BytesIO(data)
        return pd.read_excel(bio, sheet_name=sheet_name)
    else:
        return pd.read_excel(path_or_uri, sheet_name=sheet_name)

def write_yaml_any(path_or_uri, data_text: str, profile=None, region=None, content_type="text/yaml"):
    if is_s3_uri(path_or_uri):
        bucket, key = parse_s3_uri(path_or_uri)
        s3 = s3_client(profile, region)
        s3.put_object(Bucket=bucket, Key=key, Body=data_text.encode("utf-8"), ContentType=content_type)
    else:
        with open(path_or_uri, "w", encoding="utf-8") as f:
            f.write(data_text)

def is_filled(x):
    import pandas as pd
    return not (pd.isna(x) or (isinstance(x, str) and x.strip() == ""))

def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s[:60]

def detect_number(s: str):
    import re
    try:
        if re.fullmatch(r'[+-]?\d+', s):
            return int(s)
        if re.fullmatch(r'[+-]?\d+\.\d+', s):
            return float(s)
        return None
    except Exception:
        return None

def infer_topics(text: str):
    tset = set()
    lt = text.lower()
    for pat, topic in TOPIC_RULES:
        if re.search(pat, lt):
            tset.add(topic)
    return sorted(tset)

def resolve_column(df, prefer_name, contains_hint):
    # Try exact, then case-insensitive, then contains
    cols = list(df.columns)
    if prefer_name in cols:
        return prefer_name
    lower_map = {str(c).lower(): c for c in cols}
    if prefer_name.lower() in lower_map:
        return lower_map[prefer_name.lower()]
    for c in cols:
        if contains_hint.lower() in str(c).lower():
            return c
    return None

def main():
    import argparse, yaml
    from datetime import datetime

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input Excel path or s3://bucket/canonical-answers.xlsx")
    ap.add_argument("--out", dest="out_path", required=True, help="Output YAML path or s3://bucket/answers.yaml")
    ap.add_argument("--sheet", dest="sheet", default="Sheet1", help="Worksheet name (Sheet1)")
    ap.add_argument("--question-col", default="Question")
    ap.add_argument("--answer-col", default="Answer")
    ap.add_argument("--comment-col", default="Comments")
    ap.add_argument("--ref-col", default="Ref")
    ap.add_argument("--topics", action="store_true", help="Auto-tag topics using keyword rules")
    ap.add_argument("--approve", action="store_true", help="Set approval_status=approved (default draft)")
    ap.add_argument("--profile", default=None, help="AWS profile name for S3 access (optional)")
    ap.add_argument("--region", default=None, help="AWS region for S3 access (optional)")
    args = ap.parse_args()

    # Load Excel (local or S3)
    try:
        xls = read_excel_any(args.in_path, sheet_name=None, profile=args.profile, region=args.region)
    except Exception as e:
        print(f"ERROR: cannot open Excel from {args.in_path}: {e}", file=sys.stderr); sys.exit(1)

    # Choose sheet
    sheet_name = args.sheet

    # Load target sheet
    if isinstance(xls, dict):
        df = xls[sheet_name]
    else:
        df = read_excel_any(args.in_path, sheet_name=sheet_name, profile=args.profile, region=args.region)

    df.columns = [str(c).strip() for c in df.columns]

    qcol = resolve_column(df, args.question_col, "question")
    acol = resolve_column(df, args.answer_col, "answer")
    ccol = resolve_column(df, args.comment_col, "comment")
    rcol = resolve_column(df, args.ref_col, "ref")

    if not qcol or not acol:
        print(f"ERROR: Could not find question/answer columns in sheet '{sheet_name}'. Columns: {df.columns.tolist()}", file=sys.stderr)
        sys.exit(2)

    keep = df[df[qcol].apply(is_filled) & df[acol].apply(is_filled)].copy()

    records = []
    today = datetime.utcnow().date().isoformat()

    for _, row in keep.iterrows():
        q = str(row[qcol]).strip()
        a_raw = str(row[acol]).strip()
        comment = str(row[ccol]).strip() if ccol and is_filled(row.get(ccol)) else ""
        ref = str(row[rcol]).strip() if rcol and is_filled(row.get(rcol)) else ""

        lower = a_raw.lower()
        if lower in ("y", "yes", "true", "1"):
            answer_type = "Yes"
        elif lower in ("n", "no", "false", "0"):
            answer_type = "No"
        else:
            num = detect_number(a_raw)
            answer_type = num if num is not None else a_raw

        if ref:
            cid = f"q_{slugify(ref)}"
        else:
            cid = f"q_{slugify(' '.join(q.split()[:6]))}"

        topics = infer_topics(q) if args.topics else []

        rec = {
            "id": cid,
            "canonical_question": q,
            "topics": topics,
            "default": {
                "answer_type": answer_type,
                "short_comment": comment,
                "policy_snippet_id": None
            },
            "approval_status": "approved" if args.approve else "draft",
            "last_reviewed": today if args.approve else None
        }
        if ref:
            rec["source_ref"] = ref

        records.append(rec)

    records.sort(key=lambda r: (r.get("source_ref") or "", r["id"]))

    yaml_text = yaml.safe_dump(records, sort_keys=False, allow_unicode=True)

    # Write YAML (local or S3)
    write_yaml_any(args.out_path, yaml_text, profile=args.profile, region=args.region)

    print(f"Wrote {len(records)} records to {args.out_path}")
    print(f"Sheet: {sheet_name} | Columns: Q='{qcol}' A='{acol}' C='{ccol}' Ref='{rcol}'")
    if is_s3_uri(args.out_path):
        print("S3 object content-type: text/yaml")
        print("Done.")
    else:
        print("Local file written.")
    
if __name__ == "__main__":
    main()

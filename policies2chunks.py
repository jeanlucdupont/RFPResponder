#!/usr/bin/env python3
import os, io, re, json, argparse, urllib.parse
import boto3
from docx import Document

def chunk_text(text, max_words=120):
    """Split long text into ~max_words chunks."""
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

def parse_docx_stream(file_like, source_name, nda_tier="under-nda", max_words=120):
    """Yield JSON-serializable chunk dicts from a .docx bytes stream."""
    doc = Document(file_like)
    current_section = None
    for para in doc.paragraphs:
        t = para.text.strip()
        if not t:
            continue
        # Detect section headers like "3.1 Key Management"
        if re.match(r'^\d+(\.\d+)*', t):
            current_section = t
            continue
        for c in chunk_text(t, max_words=max_words):
            yield {
                "source_doc": os.path.basename(source_name),
                "section": current_section or "General",
                "text": c,
                "nda_tier": nda_tier
            }

def parse_s3_uri(uri):
    """Split s3://bucket/key into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    parsed = urllib.parse.urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key

def iter_s3_docx_objects(s3_client, bucket, prefix):
    """Iterate S3 objects (keys) ending with .docx under prefix."""
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    while True:
        resp = s3_client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".docx"):
                yield key, obj["Size"]
        if resp.get("IsTruncated"):
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        else:
            break

def upload_jsonl_stream_to_s3(s3_client, bucket, key, records_iter):
    """Upload JSONL data directly to S3 (small/medium output)."""
    buf = io.BytesIO()
    for rec in records_iter:
        if isinstance(rec, str):
            rec = rec.encode("utf-8")
        buf.write(rec)
    buf.seek(0)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue(),
        ContentType="application/json; charset=utf-8"
    )

def main():
    ap = argparse.ArgumentParser(description="Chunk policy .docx files from S3 and write JSONL back to S3.")
    ap.add_argument("--in", required=True, help="Input S3 URI prefix (e.g., s3://mybucket/policies/)")
    ap.add_argument("--out", required=True, help="Output S3 URI (e.g., s3://mybucket/policy-chunks/policy-chunks.jsonl)")
    ap.add_argument("--nda-tier", default="under-nda", choices=["public", "under-nda", "restricted"],
                    help="Default NDA tier to tag each chunk.")
    ap.add_argument("--max-words", type=int, default=120, help="Approx max words per chunk.")
    ap.add_argument("--profile", default="", help="AWS profile name (optional).")
    args = ap.parse_args()

    session_kwargs = {}
    if args.profile:
        session_kwargs["profile_name"] = args.profile
    session = boto3.Session(**session_kwargs)
    s3 = session.client("s3")

    in_bucket, in_prefix = parse_s3_uri(args.in)
    out_bucket, out_key = parse_s3_uri(args.out)

    def records():
        total_chunks = 0
        total_files = 0
        for key, size in iter_s3_docx_objects(s3, in_bucket, in_prefix):
            total_files += 1
            obj = s3.get_object(Bucket=in_bucket, Key=key)
            with io.BytesIO(obj["Body"].read()) as bio:
                for chunk in parse_docx_stream(
                    bio, source_name=key, nda_tier=args.nda_tier, max_words=args.max_words
                ):
                    total_chunks += 1
                    yield json.dumps(chunk, ensure_ascii=False) + "\n"
        print(f"Processed {total_files} files into {total_chunks} chunks.")

    print(f"Scanning {args.in} for .docx...")
    upload_jsonl_stream_to_s3(s3, out_bucket, out_key, records())
    print(f"Uploaded JSONL to {args.out}")

if __name__ == "__main__":
    main()

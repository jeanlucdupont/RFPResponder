#!/usr/bin/env python3
import os, io, re, json, argparse
import boto3
from botocore.exceptions import ClientError
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
                "nda_tier": nda_tier  # public / under-nda / restricted
            }

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

def upload_jsonl_stream_to_s3(s3_client, bucket, key, records_iter, part_size=8*1024*1024):
    """
    Stream JSONL records to S3 using multipart upload.
    records_iter must yield bytes or str (one JSON line at a time).
    """
    # Start multipart upload
    mpu = s3_client.create_multipart_upload(
        Bucket=bucket,
        Key=key,
        ContentType="application/json; charset=utf-8"
    )
    upload_id = mpu["UploadId"]
    parts = []
    part_number = 1
    buffer = io.BytesIO()

    def _flush_part():
        nonlocal part_number, buffer
        buffer.seek(0)
        if buffer.tell() == 0 and buffer.getbuffer().nbytes == 0:
            return None
        resp = s3_client.upload_part(
            Bucket=bucket,
            Key=key,
            PartNumber=part_number,
            UploadId=upload_id,
            Body=buffer
        )
        parts.append({"ETag": resp["ETag"], "PartNumber": part_number})
        part_number += 1
        buffer = io.BytesIO()

    try:
        for rec in records_iter:
            if isinstance(rec, str):
                rec = rec.encode("utf-8")
            # If next record won't fit, flush current part
            if buffer.tell() + len(rec) > part_size:
                _flush_part()
            buffer.write(rec)
        # Flush any remaining data
        if buffer.tell() > 0:
            _flush_part()

        s3_client.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts}
        )
    except Exception as e:
        # Abort MPU on error to avoid leaking parts
        s3_client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        raise e

def main():
    ap = argparse.ArgumentParser(description="Chunk policy .docx files from S3 and write JSONL back to S3.")
    ap.add_argument("--bucket", required=True, help="S3 bucket containing input policies and for output JSONL.")
    ap.add_argument("--input-prefix", required=True, help="S3 prefix where .docx policies live (e.g., policies/).")
    ap.add_argument("--output-key", default="policy-chunks/policy-chunks.jsonl",
                    help="S3 key for output JSONL (e.g., policy-chunks/policy-chunks.jsonl).")
    ap.add_argument("--profile", default="", help="AWS profile name (optional).")
    ap.add_argument("--region", default=None, help="AWS region name (optional).")
    ap.add_argument("--nda-tier", default="under-nda", choices=["public", "under-nda", "restricted"],
                    help="Default NDA tier to tag each chunk.")
    ap.add_argument("--max-words", type=int, default=120, help="Approx max words per chunk.")
    args = ap.parse_args()

    session_kwargs = {}
    if args.profile:
        session_kwargs["profile_name"] = args.profile
    if args.region:
        session_kwargs["region_name"] = args.region

    session = boto3.Session(**session_kwargs)
    s3 = session.client("s3")

    # Record generator: lazily list, download, parse, and yield JSONL lines
    def records():
        total_chunks = 0
        total_files = 0
        for key, size in iter_s3_docx_objects(s3, args.bucket, args.input_prefix):
            total_files += 1
            obj = s3.get_object(Bucket=args.bucket, Key=key)
            with io.BytesIO(obj["Body"].read()) as bio:
                for chunk in parse_docx_stream(
                    bio, source_name=key, nda_tier=args.nda-tier if hasattr(args, "nda-tier") else args.nda_tier,
                    max_words=args.max_words
                ):
                    total_chunks += 1
                    yield (json.dumps(chunk, ensure_ascii=False) + "\n")
        # Emit a short progress line to STDOUT at the end (not part of JSONL)
        print(f"Processed {total_files} files into {total_chunks} chunks.")

    print(f"Scanning s3://{args.bucket}/{args.input-prefix if hasattr(args,'input-prefix') else args.input_prefix} for .docx...")
    upload_jsonl_stream_to_s3(s3, args.bucket, args.output_key, records_iter=records())
    print(f"Uploaded JSONL to s3://{args.bucket}/{args.output_key}")

if __name__ == "__main__":
    main()

import os, argparse, json, re
import boto3
from docx import Document

def chunk_text(text, max_words=120):
    """Split long text into ~max_words chunks"""
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

def parse_docx(path):
    doc = Document(path)
    chunks = []
    current_section = None
    for para in doc.paragraphs:
        t = para.text.strip()
        if not t: 
            continue
        # detect section headers like "3.1 Key Management"
        if re.match(r'^\d+(\.\d+)*', t):
            current_section = t
            continue
        for c in chunk_text(t):
            chunks.append({
                "source_doc": os.path.basename(path),
                "section": current_section or "General",
                "text": c,
                "nda_tier": "under-nda"   # adjust: public / under-nda / restricted
            })
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Local folder with .docx policies")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="policy-chunks/")
    ap.add_argument("--profile", default="")
    args = ap.parse_args()

    session = boto3.Session(profile_name=args.profile)
    s3 = session.client("s3")

    all_chunks = []
    for f in os.listdir(args.in_dir):
        if f.lower().endswith(".docx"):
            path = os.path.join(args.in_dir, f)
            chunks = parse_docx(path)
            all_chunks.extend(chunks)

    # Write local JSONL
    out_file = "policy-chunks.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_chunks)} chunks to {out_file}")

    # Upload to S3
    s3.upload_file(out_file, args.bucket, f"{args.prefix}policy-chunks.jsonl")
    print(f"Uploaded to s3://{args.bucket}/{args.prefix}policy-chunks.jsonl")

if __name__ == "__main__":
    main()

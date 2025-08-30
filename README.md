# RFPResponder

**RFPResponder** is an AWS-powered framework to **automatically draft answers to security questionnaires and RFPs** by reusing canonical answers and mapping them to policy excerpts.

Instead of manually filling in dozens of repetitive *Yes / No / Number* answers, RFPResponder lets you:
- Define **canonical answers** to common RFP questions (with optional policy references).
- Parse and chunk your **security policies** into searchable snippets.
- Use **Amazon Bedrock Titan embeddings** to auto-link questions to the right policies.
- Feed in a new Excel questionnaire and generate a pre-filled spreadsheet.

---

## Disclaimer

This is a proof of concept, built for experimentation in a constrained environment.  
In a real-world setting, many additional parameters and safeguards would be required.

The dataset used here is intentionally low-quality:  
- A questionnaire from 2012 based on a standard that no longer exists.  
- Generic policies sourced from SANS.  

For meaningful results in production, **real, current data** must be ingested (your organization’s actual policies and questionnaires). That would produce far more accurate and useful mappings.

---

## Architecture Overview

1. **Canonical answers**  
   - Input: `inbox/canonical-answers.xlsx` (Excel file containing all standard questions and answers).  
   - Process: `canonical2yaml.py` parses the file and generates a YAML version.  
   - Output: `library/answers.yaml`.

2. **Policy chunks**  
   - Input: `policies/` (directory of Word-based cybersecurity policies and procedures).  
   - Process: `policies2chunks.py` parses and chunks policy text into manageable snippets.  
   - Output: `library/policychunks.json`.

3. **Linking canonicals to policy chunks**  
   - Process: `genlib.py` uses **Amazon Titan embeddings** to link canonical answers with relevant policy snippets.  
   - Inputs: `library/answers.yaml` + `library/policychunks.json`.  
   - Outputs:  
     - `library/standard-answers.yaml` (enriched canonical answers with policy references).  
     - `proposed_links.yaml` (candidate matches with confidence scores).

4. **Questionnaire answering**  
   - Process: `rfp-responder.py` consumes a new Excel questionnaire and auto-fills suggested answers, comments, and references.  
   - Inputs: `inbox/questionnaire.xlsx` + `library/standard-answers.yaml`.  
   - Output: `output/questionnaire_filled.xlsx`.

---

## Script Usage

**canonical2yaml.py**
```bash
python canonical2yaml.py `
  --in s3://my-bucket/inbox/canonical-answers.xlsx `
  --out s3://my-bucket/library/answers.yaml `
  --topics --approve --profile XXXXXX
```

**policies2chunks.py**
```bash
python policies2chunks.py `
  --in s3://my-bucket/policies/ `
  --out s3://my-bucket/library/policychunks.json `
  --profile XXXXXX
```

**genlib.py**
```bash
python genlib.py `
  --answers s3://my-bucket/library/answers.yaml `
  --chunks  s3://my-bucket/library/policychunks.json `
  --out     s3://my-bucket/library/standard-answers.yaml `
  --min-sim 0.84 --topk 3 `
  --region us-east-1 `
  --profile myprofile
```

**rfp-responder.py**
```bash
python rfp-responder.py `
  --xlsx s3://my-bucket/inbox/questionnaire.xlsx `
  --yaml s3://my-bucket/library/standard-answers.yaml `
  --out  s3://my-bucket/output/questionnaire_filled.xlsx `
  --region us-east-1 `
  --min-sim 0.78 `
  --profile myprofile 
```

---

## Workflow Diagram

```mermaid
flowchart TD
  subgraph Inputs
    A[Canonical Answers (Excel)<br/>inbox/canonical-answers.xlsx]
    B[Policies (.docx)<br/>policies/]
    C[New Questionnaire (Excel)<br/>inbox/questionnaire.xlsx]
  end

  subgraph Processing
    S1[canonical2yaml.py<br/>(--topics --approve)]
    S2[policies2chunks.py]
    S3[genlib.py<br/>Amazon Bedrock Titan Embeddings<br/>(--min-sim --topk)]
    S4[rfp-responder.py<br/>(--min-sim)]
  end

  subgraph Library
    L1[library/answers.yaml]
    L2[library/policychunks.json]
    L3[library/standard-answers.yaml]
    L4[proposed_links.yaml<br/>(candidates + confidence)]
  end

  subgraph Output
    O1[output/questionnaire_filled.xlsx]
  end

  A --> S1 --> L1
  B --> S2 --> L2
  L1 --> S3
  L2 --> S3
  S3 --> L3
  S3 --> L4
  C --> S4
  L3 --> S4
  S4 --> O1

  note right of S3: Uses vector similarity to link<br/>canonical answers to policy chunks

  classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:1px,color:#0d47a1;
  classDef process fill:#fff8e1,stroke:#f9a825,stroke-width:1px,color:#6d4c41;
  classDef lib fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px,color:#1b5e20;
  classDef output fill:#f3e5f5,stroke:#6a1b9a,stroke-width:1px,color:#4a148c;

  class A,B,C input
  class S1,S2,S3,S4 process
  class L1,L2,L3,L4 lib
  class O1 output
```

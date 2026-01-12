# Code Explanation - PDF to Vector Database Pipeline

## Overview

This pipeline extracts text from PDF files, converts it into vector embeddings using OpenAI's embedding models, and stores the data in a PostgreSQL database with pgvector extension for semantic search capabilities.

---

## Code Structure Explained

### 1. Imports and Setup (Lines 1-7)

```1:7:pgvector_pipeline.py
import os
from dotenv import load_dotenv
import psycopg2
from pypdf import PdfReader
from openai import OpenAI

load_dotenv()
```

**What it does:**
- Imports necessary libraries: `os` for file paths, `dotenv` for environment variables, `psycopg2` for PostgreSQL database connection, `pypdf` for reading PDF files, and `openai` for creating embeddings
- `load_dotenv()` loads configuration from a `.env` file into environment variables

---

### 2. Configuration (Lines 9-22)

```9:22:pgvector_pipeline.py
# ---- config (strict) ----
DATABASE_URL = os.environ["DATABASE_URL"]

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_EMBED_MODEL = os.environ["OPENAI_EMBED_MODEL"]
EMBED_DIM = int(os.environ["EMBED_DIM"])

PDFS = [os.environ["PDF_1"], os.environ["PDF_2"]]

TABLE = "insurance_policy_data"

# Chunking (smaller chunks = better retrieval)
CHUNK_SIZE = 1200
OVERLAP = 200
```

**What it does:**
- Reads configuration from environment variables:
  - `DATABASE_URL`: PostgreSQL connection string (e.g., `postgresql://user:password@host:port/database`)
  - `OPENAI_API_KEY`: API key for OpenAI embeddings
  - `OPENAI_EMBED_MODEL`: Name of the embedding model (e.g., `text-embedding-3-small`)
  - `EMBED_DIM`: Dimension of embedding vectors (e.g., 1536 for `text-embedding-3-small`)
  - `PDF_1` and `PDF_2`: Paths to PDF files to process
- Sets constants:
  - `TABLE`: Database table name
  - `CHUNK_SIZE`: Number of characters per text chunk (1200)
  - `OVERLAP`: Number of overlapping characters between chunks (200) - helps maintain context

---

### 3. Helper Functions

#### `read_pdf_pages()` - Extract Text from PDF (Lines 25-31)

```25:31:pgvector_pipeline.py
def read_pdf_pages(path: str):
    """Return list of (page_num, page_text). page_num is 1-based."""
    reader = PdfReader(path)
    pages = []
    for i, p in enumerate(reader.pages, start=1):
        pages.append((i, (p.extract_text() or "").strip()))
    return pages
```

**What it does:**
- Takes a PDF file path as input
- Reads the PDF and extracts text from each page
- Returns a list of tuples: `(page_number, page_text)`
- Page numbers start at 1 (1-based indexing)
- Empty pages return empty strings (not skipped here, but handled later)

---

#### `chunk_page_text()` - Split Text into Chunks (Lines 34-46)

```34:46:pgvector_pipeline.py
def chunk_page_text(page_text: str):
    """Chunk a single page with overlap (character-based, but per-page)."""
    chunks = []
    start = 0
    while start < len(page_text):
        end = min(start + CHUNK_SIZE, len(page_text))
        chunk = page_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(page_text):
            break
        start = end - OVERLAP
    return chunks
```

**What it does:**
- Splits a page's text into smaller chunks of `CHUNK_SIZE` characters (1200)
- Uses overlapping chunks: each new chunk starts `OVERLAP` characters (200) before the previous chunk ended
- This overlap ensures context isn't lost at chunk boundaries
- Returns a list of text chunks (non-empty strings only)

**Example:** If text is 2400 characters:
- Chunk 1: characters 0-1200
- Chunk 2: characters 1000-2200 (200 char overlap)
- Chunk 3: characters 2000-2400 (200 char overlap)

---

#### `to_vec_str()` - Convert Vector to String (Lines 49-50)

```49:50:pgvector_pipeline.py
def to_vec_str(vec):
    return "[" + ",".join(str(x) for x in vec) + "]"
```

**What it does:**
- Converts a Python list of numbers (embedding vector) into a string format
- Format: `[0.123,-0.456,0.789,...]`
- Required for storing vectors in PostgreSQL with pgvector extension

---

### 4. Main Function - The Pipeline

#### Setup: Initialize Clients and Database (Lines 53-75)

```53:75:pgvector_pipeline.py
def main():
    # Clients
    oai = OpenAI(api_key=OPENAI_API_KEY)

    # DB
    conn = psycopg2.connect(DATABASE_URL)

    # Create schema (includes page_num)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id bigserial PRIMARY KEY,
            doc text NOT NULL,
            page_num int NOT NULL,
            chunk_id int NOT NULL,
            content text NOT NULL,
            embedding vector({EMBED_DIM}) NOT NULL,
            UNIQUE (doc, chunk_id)
        );
    """)
    conn.commit()
    cur.close()
```

**What it does:**
- Creates OpenAI client for embedding API calls
- Connects to PostgreSQL database using the connection string
- Enables pgvector extension (allows storing vector data types)
- Creates the database table if it doesn't exist with columns:
  - `id`: Auto-incrementing primary key
  - `doc`: Document name (PDF filename)
  - `page_num`: Page number where chunk came from
  - `chunk_id`: Unique identifier for chunk within document
  - `content`: The actual text content of the chunk
  - `embedding`: Vector embedding (array of numbers representing text semantically)
- `UNIQUE (doc, chunk_id)` ensures no duplicate chunks per document

---

#### PDF Processing Loop (Lines 77-123)

```77:123:pgvector_pipeline.py
    # Ingest PDFs
    for path in PDFS:
        doc = os.path.basename(path)
        print(f"\nReading {doc}...")

        pages = read_pdf_pages(path)

        # Build (page_num, chunk_text) list
        chunk_rows = []
        for page_num, page_text in pages:
            if not page_text:
                continue
            for ch in chunk_page_text(page_text):
                chunk_rows.append((page_num, ch))

        if not chunk_rows:
            print("No text extracted, skipping.")
            continue

        print(f"Chunks: {len(chunk_rows)}")

        # Embed with small prefix (improves retrieval)
        embed_inputs = [
            f"Document: {doc}\nPage: {page_num}\nText:\n{chunk}"
            for (page_num, chunk) in chunk_rows
        ]

        print("Embedding (OpenAI)...")
        emb = oai.embeddings.create(model=OPENAI_EMBED_MODEL, input=embed_inputs)
        vectors = [x.embedding for x in emb.data]

        # Insert (idempotent: updates if rerun)
        cur = conn.cursor()
        for i, ((page_num, chunk), vec) in enumerate(zip(chunk_rows, vectors)):
            cur.execute(f"""
                INSERT INTO {TABLE} (doc, page_num, chunk_id, content, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                ON CONFLICT (doc, chunk_id)
                DO UPDATE SET
                    page_num = EXCLUDED.page_num,
                    content  = EXCLUDED.content,
                    embedding = EXCLUDED.embedding;
            """, (doc, page_num, i, chunk, to_vec_str(vec)))
        conn.commit()
        cur.close()

        print(f"Inserted/Updated {doc} ✅")
```

**What it does (per PDF file):**

1. **Extract filename** from the full path
2. **Read PDF pages** using `read_pdf_pages()` function
3. **Chunk the text**: For each page, split text into overlapping chunks and collect them with page numbers
4. **Skip empty documents**: If no text was extracted, move to next PDF
5. **Create embeddings**: 
   - Format each chunk with document name and page number (helps AI understand context)
   - Send all chunks to OpenAI embedding API in one batch
   - Extract the vector embeddings from the response
6. **Store in database**:
   - For each chunk-embedding pair, insert into database
   - Uses `ON CONFLICT` to update existing records if rerun (idempotent - safe to run multiple times)
   - Commits all inserts/updates for the document
7. **Print success message**

---

#### Cleanup (Lines 125-129)

```125:129:pgvector_pipeline.py
    conn.close()


if __name__ == "__main__":
    main()
```

**What it does:**
- Closes database connection after processing all PDFs
- Entry point: Only runs `main()` when script is executed directly (not when imported)

---

## Visual Flow Diagrams

### Overall Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    PDF INGESTION PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

STEP 1: CONFIGURATION
═══════════════════════════════════════
    .env file                    Configuration Variables
    ├─ DATABASE_URL      ──────► DATABASE_URL
    ├─ OPENAI_API_KEY    ──────► OPENAI_API_KEY
    ├─ OPENAI_EMBED_MODEL──────► OPENAI_EMBED_MODEL
    ├─ EMBED_DIM         ──────► EMBED_DIM (e.g., 1536)
    ├─ PDF_1             ──────► PDFS[0]
    └─ PDF_2             ──────► PDFS[1]


STEP 2: DATABASE INITIALIZATION
═══════════════════════════════════════
    PostgreSQL Database
    │
    ├─ CREATE EXTENSION vector  (Enable vector support)
    │
    └─ CREATE TABLE insurance_policy_data
         ├─ id (auto-increment)
         ├─ doc (document name)
         ├─ page_num (page number)
         ├─ chunk_id (chunk identifier)
         ├─ content (text content)
         └─ embedding (vector) ← Stores numerical representation


STEP 3: PDF PROCESSING (For each PDF)
═══════════════════════════════════════

    PDF File (AUTO-POLICY.pdf)
         │
         ▼
    ┌─────────────────┐
    │ read_pdf_pages()│
    └─────────────────┘
         │
         ▼
    Extract Pages
    ├─ Page 1: "Coverage details..."
    ├─ Page 2: "Premium information..."
    ├─ Page 3: "Terms and conditions..."
    └─ ...
         │
         ▼
    ┌──────────────────┐
    │ chunk_page_text()│
    └──────────────────┘
         │
         ▼
    Split into Chunks (with overlap)
    ├─ Chunk 1: [chars 0-1200]
    ├─ Chunk 2: [chars 1000-2200]  ← 200 char overlap
    ├─ Chunk 3: [chars 2000-3200]  ← 200 char overlap
    └─ ...


STEP 4: EMBEDDING CREATION
═══════════════════════════════════════

    Text Chunks
         │
         ▼
    Format with context:
    "Document: AUTO-POLICY.pdf
     Page: 1
     Text: [chunk content]"
         │
         ▼
    ┌──────────────────────┐
    │  OpenAI Embeddings   │
    │  API (batch request) │
    └──────────────────────┘
         │
         ▼
    Vector Embeddings (numbers)
    Chunk 1: [0.123, -0.456, 0.789, ...]  (1536 numbers)
    Chunk 2: [0.234, -0.567, 0.890, ...]
    Chunk 3: [0.345, -0.678, 0.901, ...]
    └─ ...


STEP 5: DATABASE STORAGE
═══════════════════════════════════════

    (Chunk + Embedding) Pairs
         │
         ▼
    ┌──────────────────────┐
    │  PostgreSQL Database │
    │  INSERT/UPDATE       │
    │  (idempotent)        │
    └──────────────────────┘
         │
         ▼
    Table: insurance_policy_data
    ┌────┬──────────┬──────────┬──────────┬─────────────┬────────────────────┐
    │ id │ doc      │ page_num │ chunk_id │ content     │ embedding          │
    ├────┼──────────┼──────────┼──────────┼─────────────┼────────────────────┤
    │ 1  │ AUTO...  │    1     │    0     │ "Coverage..."│ [0.123, -0.456...]│
    │ 2  │ AUTO...  │    1     │    1     │ "details..." │ [0.234, -0.567...]│
    │ 3  │ HOME...  │    1     │    0     │ "Policy..."  │ [0.345, -0.678...]│
    └────┴──────────┴──────────┴──────────┴─────────────┴────────────────────┘
```

### Data Flow Diagram

```
INPUT FILES
    │
    ├─── PDF_1 (AUTO-POLICY.pdf)
    │        │
    │        ▼
    │    [Extract Text] ──┐
    │        │            │
    │        ▼            │
    │    [Split into Chunks] ──┐
    │        │                  │
    │        ▼                  │
    │    [Create Embeddings] ───┼──┐
    │        │                  │  │
    │        ▼                  │  │
    │    [Store in DB] ─────────┼──┼──┐
    │                           │  │  │
    └─── PDF_2 (HOMEOWNERS-POLICY.pdf)
             │                  │  │  │
             ▼                  │  │  │
         [Extract Text] ──┐     │  │  │
             │            │     │  │  │
             ▼            │     │  │  │
         [Split into Chunks] ───┘  │  │
             │                     │  │
             ▼                     │  │
         [Create Embeddings] ──────┘  │
             │                        │
             ▼                        │
         [Store in DB] ───────────────┘
                                      │
                                    DONE ✅
```

### Chunking Visualization

```
Original Page Text (2400 characters)
═══════════════════════════════════════════════════════════════════
│                                                                │
│  "This is the insurance policy text. It contains information  │
│   about coverage, premiums, and terms. The coverage includes  │
│   comprehensive protection for your vehicle. Premiums are      │
│   calculated based on various factors. Terms and conditions   │
│   apply to all policies. Additional coverage options are      │
│   available for purchase. Discounts may apply for safe        │
│   drivers. Claims process is straightforward. Contact your    │
│   agent for more information. Policy renewal is automatic..." │
│                                                                │
═══════════════════════════════════════════════════════════════════

After Chunking (CHUNK_SIZE=1200, OVERLAP=200):

Chunk 0: [0 ──────────────── 1200] 
         │                    │
         "This is the insurance policy text. It contains information
          about coverage, premiums, and terms. The coverage includes
          comprehensive protection for your vehicle. Premiums are
          calculated based on various factors. Terms and conditions..."

Chunk 1:        [1000 ──────────────── 2200]
                │                    │
                "factors. Terms and conditions apply to all policies.
                 Additional coverage options are available for purchase.
                 Discounts may apply for safe drivers. Claims process
                 is straightforward. Contact your agent for more..."

Chunk 2:                      [2000 ──────────────── 2400]
                              │                    │
                              "information. Policy renewal is automatic..."

Key: 
├─ Chunk boundaries
└─ Overlap region (200 chars shared between chunks)
```

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ENVIRONMENT                              │
│  .env file with API keys, database URL, file paths              │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PYTHON APPLICATION                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  PDF Reader  │  │  Text        │  │  Embedding   │          │
│  │  (pypdf)     │─►│  Chunker     │─►│  Creator     │          │
│  └──────────────┘  └──────────────┘  │  (OpenAI)    │          │
│                                       └──────────────┘          │
│                                              │                  │
│                                              ▼                  │
│                                       ┌──────────────┐          │
│                                       │  PostgreSQL  │          │
│                                       │  + pgvector  │          │
│                                       └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
         │                               │
         ▼                               ▼
┌─────────────────┐              ┌──────────────┐
│  PDF Files      │              │  Database    │
│  (data/)        │              │  (PostgreSQL)│
│                 │              │              │
│  • AUTO-POLICY  │              │  • Tables    │
│  • HOMEOWNERS   │              │  • Vectors   │
└─────────────────┘              └──────────────┘
```

---

## Key Concepts

### Embeddings
- **What**: Numerical representations of text that capture semantic meaning
- **Why**: Similar texts have similar embeddings, enabling semantic search
- **Example**: "car insurance" and "vehicle coverage" have similar embeddings even with different words
- **Format**: Array of numbers (e.g., 1536 numbers for `text-embedding-3-small`)

### Chunking with Overlap
- **Why chunk?**: Embeddings work better with smaller, focused pieces of text
- **Why overlap?**: Prevents loss of context at boundaries (sentences that span chunks)
- **Example**: If a sentence ends at character 1195 and next starts at 1205, overlap ensures both chunks include the full context

### Idempotent Operations
- **What**: The script can be run multiple times safely
- **How**: Uses `ON CONFLICT DO UPDATE` SQL clause
- **Benefit**: If you add new PDFs or update existing ones, rerunning updates the database without creating duplicates

### Database Schema
- **Primary Key**: `id` - unique identifier for each row
- **Unique Constraint**: `(doc, chunk_id)` - ensures each chunk appears only once per document
- **Vector Column**: `embedding vector(1536)` - stores the numerical embedding (dimension matches your `EMBED_DIM` setting)

---

## Common Configuration Values

### OpenAI Embedding Models

| Model Name | Default Dimension | Use Case |
|------------|-------------------|----------|
| `text-embedding-3-small` | 1536 | Good balance of quality and cost |
| `text-embedding-3-large` | 3072 | Higher quality, larger storage |
| `text-embedding-ada-002` | 1536 | Older model, still reliable |

### Example .env Configuration

```bash
DATABASE_URL=postgresql://user:password@localhost:5432/vectordb
OPENAI_API_KEY=sk-...
OPENAI_EMBED_MODEL=text-embedding-3-small
EMBED_DIM=1536
PDF_1=data/AUTO-POLICY.pdf
PDF_2=data/HOMEOWNERS-POLICY.pdf
```

**Important**: The `EMBED_DIM` must match the dimension of your chosen embedding model!

---

## Summary

This pipeline performs the following steps:

1. **Reads** PDF files from specified paths
2. **Extracts** text page by page
3. **Splits** text into smaller chunks with overlapping boundaries
4. **Converts** text chunks to numerical vectors (embeddings) using OpenAI's API
5. **Stores** chunks and their embeddings in PostgreSQL with pgvector extension
6. **Supports** idempotent operations (safe to rerun)

The result is a searchable database where you can later perform semantic searches to find relevant content based on meaning, not just keyword matching.

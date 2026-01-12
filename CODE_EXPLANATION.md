# Code Explanation - PDF to Vector Database Pipeline

## Line-by-Line Explanation

### Import Section (Lines 1-6)
```python
import os
```
- **Line 1**: Import the `os` module - used to work with file paths and environment variables

```python
from dotenv import load_dotenv
```
- **Line 2**: Import function to load environment variables from a `.env` file

```python
import psycopg2
```
- **Line 3**: Import PostgreSQL database connector library

```python
from pypdf import PdfReader
```
- **Line 4**: Import PDF reader to extract text from PDF files

```python
from openai import OpenAI
```
- **Line 5**: Import OpenAI library to create embeddings (vector representations of text)

```python
import anthropic
```
- **Line 6**: Import Anthropic library to use Claude AI for generating answers

### Environment Setup (Line 8)
```python
load_dotenv()
```
- **Line 8**: Load all environment variables from the `.env` file into the program

### Configuration Section (Lines 10-30)
```python
# ---- config (strict) ----
```
- **Line 10**: Comment marking the configuration section

```python
DATABASE_URL = os.environ["DATABASE_URL"]
```
- **Line 11**: Get the database connection string from environment variables (e.g., `postgresql://user:pass@host:port/db`)

```python
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
```
- **Line 13**: Get OpenAI API key for making embedding requests

```python
OPENAI_EMBED_MODEL = os.environ["OPENAI_EMBED_MODEL"]
```
- **Line 14**: Get the name of the OpenAI model to use for creating embeddings (e.g., "text-embedding-3-small")

```python
EMBED_DIM = int(os.environ["EMBED_DIM"])
```
- **Line 15**: Get the dimension (size) of the embedding vectors and convert to integer (e.g., 1536)

```python
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
```
- **Line 17**: Get Anthropic API key (using `.get()` so it won't crash if missing)

```python
CLAUDE_MODEL = os.environ["CLAUDE_MODEL"]
```
- **Line 18**: Get the Claude model name (e.g., "claude-3-sonnet-20240229")

```python
PDFS = [os.environ["PDF_1"], os.environ["PDF_2"]]
```
- **Line 20**: Create a list with paths to two PDF files from environment variables

```python
TEST_QUERY = os.environ["TEST_QUERY"]
```
- **Line 21**: Get the test question/query to search for in the documents

```python
TABLE = "insurance_policy_data"
```
- **Line 23**: Set the database table name where we'll store the data

```python
# Chunking (smaller chunks = better retrieval)
CHUNK_SIZE = 1200
```
- **Line 26**: Set how many characters each text chunk should contain (1200 chars)

```python
OVERLAP = 200
```
- **Line 27**: Set how many characters should overlap between consecutive chunks (keeps context)

```python
# Retrieval count (more context = better answers)
TOP_K = 12
```
- **Line 30**: Set how many top matching chunks to retrieve when searching (12 chunks)

### Helper Functions

#### read_pdf_pages Function (Lines 33-39)
```python
def read_pdf_pages(path: str):
```
- **Line 33**: Define a function that takes a PDF file path as input

```python
    """Return list of (page_num, page_text). page_num is 1-based."""
```
- **Line 34**: Documentation string explaining what the function returns

```python
    reader = PdfReader(path)
```
- **Line 35**: Create a PDF reader object to read the PDF file

```python
    pages = []
```
- **Line 36**: Create an empty list to store pages

```python
    for i, p in enumerate(reader.pages, start=1):
```
- **Line 37**: Loop through each page in the PDF, starting page numbers at 1

```python
        pages.append((i, (p.extract_text() or "").strip()))
```
- **Line 38**: Extract text from the page, if empty use "", strip whitespace, and add (page_num, text) to the list

```python
    return pages
```
- **Line 39**: Return the list of (page_number, page_text) pairs

#### chunk_page_text Function (Lines 42-54)
```python
def chunk_page_text(page_text: str):
```
- **Line 42**: Define a function that splits page text into smaller chunks

```python
    """Chunk a single page with overlap (character-based, but per-page)."""
```
- **Line 43**: Documentation explaining the function

```python
    chunks = []
```
- **Line 44**: Create empty list to store text chunks

```python
    start = 0
```
- **Line 45**: Set starting position in the text to 0

```python
    while start < len(page_text):
```
- **Line 46**: Loop while we haven't reached the end of the text

```python
        end = min(start + CHUNK_SIZE, len(page_text))
```
- **Line 47**: Calculate end position (either start+CHUNK_SIZE or end of text, whichever is smaller)

```python
        chunk = page_text[start:end].strip()
```
- **Line 48**: Extract the chunk text from start to end and remove extra whitespace

```python
        if chunk:
```
- **Line 49**: Check if chunk is not empty

```python
            chunks.append(chunk)
```
- **Line 50**: Add the chunk to our list

```python
        if end == len(page_text):
```
- **Line 51**: Check if we've reached the end of the text

```python
            break
```
- **Line 52**: Exit the loop if we're done

```python
        start = end - OVERLAP
```
- **Line 53**: Move start position back by OVERLAP amount (creates overlapping chunks)

```python
    return chunks
```
- **Line 54**: Return the list of text chunks

#### to_vec_str Function (Lines 57-58)
```python
def to_vec_str(vec):
```
- **Line 57**: Define a function to convert a vector (list of numbers) to a string format

```python
    return "[" + ",".join(str(x) for x in vec) + "]"
```
- **Line 58**: Convert vector to string format like "[0.1,0.2,0.3]" for database storage

### Main Function (Lines 61-183)

#### Setup (Lines 62-67)
```python
def main():
```
- **Line 61**: Define the main function that runs the entire pipeline

```python
    # Clients
    oai = OpenAI(api_key=OPENAI_API_KEY)
```
- **Line 63**: Create OpenAI client object using the API key

```python
    claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
```
- **Line 64**: Create Anthropic client object for Claude AI

```python
    # DB
    conn = psycopg2.connect(DATABASE_URL)
```
- **Line 67**: Connect to PostgreSQL database using the connection string

#### Database Schema Creation (Lines 69-84)
```python
    # Create schema (includes page_num)
    cur = conn.cursor()
```
- **Line 70**: Create a cursor object to execute SQL commands

```python
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
```
- **Line 71**: Enable the pgvector extension (allows storing vectors in PostgreSQL)

```python
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
```
- **Line 72-73**: Start creating the database table if it doesn't exist

```python
            id bigserial PRIMARY KEY,
```
- **Line 74**: Create an auto-incrementing ID column as the primary key

```python
            doc text NOT NULL,
```
- **Line 75**: Column to store the document name (PDF filename)

```python
            page_num int NOT NULL,
```
- **Line 76**: Column to store which page number the chunk came from

```python
            chunk_id int NOT NULL,
```
- **Line 77**: Column to store which chunk number it is within the document

```python
            content text NOT NULL,
```
- **Line 78**: Column to store the actual text content of the chunk

```python
            embedding vector({EMBED_DIM}) NOT NULL,
```
- **Line 79**: Column to store the vector embedding (list of numbers representing the text)

```python
            UNIQUE (doc, chunk_id)
```
- **Line 80**: Ensure each document-chunk combination is unique

```python
        );
    """)
```
- **Line 81-82**: Close the table definition

```python
    conn.commit()
```
- **Line 83**: Save the table creation to the database

```python
    cur.close()
```
- **Line 84**: Close the cursor

#### PDF Ingestion Loop (Lines 86-132)
```python
    # Ingest PDFs
    for path in PDFS:
```
- **Line 87**: Loop through each PDF file path

```python
        doc = os.path.basename(path)
```
- **Line 88**: Extract just the filename from the full path (e.g., "policy.pdf")

```python
        print(f"\nReading {doc}...")
```
- **Line 89**: Print a message showing which document is being processed

```python
        pages = read_pdf_pages(path)
```
- **Line 91**: Read all pages from the PDF file

```python
        # Build (page_num, chunk_text) list
        chunk_rows = []
```
- **Line 94**: Create empty list to store chunk data

```python
        for page_num, page_text in pages:
```
- **Line 95**: Loop through each page

```python
            if not page_text:
```
- **Line 96**: Check if page is empty

```python
                continue
```
- **Line 97**: Skip empty pages

```python
            for ch in chunk_page_text(page_text):
```
- **Line 98**: Loop through each chunk in the page

```python
                chunk_rows.append((page_num, ch))
```
- **Line 99**: Add (page_number, chunk_text) to our list

```python
        if not chunk_rows:
```
- **Line 101**: Check if no chunks were created

```python
            print("No text extracted, skipping.")
            continue
```
- **Line 102-103**: Print message and skip to next PDF if no text found

```python
        print(f"Chunks: {len(chunk_rows)}")
```
- **Line 105**: Print how many chunks were created

```python
        # Embed with small prefix (improves retrieval)
        embed_inputs = [
```
- **Line 108**: Start creating list of texts to embed

```python
            f"Document: {doc}\nPage: {page_num}\nText:\n{chunk}"
            for (page_num, chunk) in chunk_rows
```
- **Line 109-110**: Create formatted text for each chunk with document name and page number (helps AI understand context)

```python
        ]
```
- **Line 111**: Close the list

```python
        print("Embedding (OpenAI)...")
```
- **Line 113**: Print status message

```python
        emb = oai.embeddings.create(model=OPENAI_EMBED_MODEL, input=embed_inputs)
```
- **Line 114**: Send all chunks to OpenAI to create embeddings (vector representations)

```python
        vectors = [x.embedding for x in emb.data]
```
- **Line 115**: Extract just the embedding vectors from the response

```python
        # Insert (idempotent: updates if rerun)
        cur = conn.cursor()
```
- **Line 118**: Create a new cursor for database operations

```python
        for i, ((page_num, chunk), vec) in enumerate(zip(chunk_rows, vectors)):
```
- **Line 119**: Loop through chunks and their corresponding vectors, with index `i`

```python
            cur.execute(f"""
                INSERT INTO {TABLE} (doc, page_num, chunk_id, content, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
```
- **Line 121-122**: SQL to insert a row into the table

```python
                ON CONFLICT (doc, chunk_id)
                DO UPDATE SET
```
- **Line 123-124**: If the same document-chunk already exists, update it instead

```python
                    page_num = EXCLUDED.page_num,
                    content  = EXCLUDED.content,
                    embedding = EXCLUDED.embedding;
```
- **Line 125-127**: Update the existing row with new values

```python
            """, (doc, page_num, i, chunk, to_vec_str(vec)))
```
- **Line 128**: Provide the values for the SQL query (safe from SQL injection)

```python
        conn.commit()
```
- **Line 129**: Save all the inserts/updates to the database

```python
        cur.close()
```
- **Line 130**: Close the cursor

```python
        print(f"Inserted/Updated {doc} ✅")
```
- **Line 132**: Print success message

#### Vector Search (Lines 134-150)
```python
    # Vector search
    print("\nSearching...")
```
- **Line 135**: Print status message

```python
    # Embed query with a hint
    query_text = f"Insurance policy question:\n{TEST_QUERY}"
```
- **Line 138**: Format the user's question with a prefix for better matching

```python
    qvec = oai.embeddings.create(model=OPENAI_EMBED_MODEL, input=[query_text]).data[0].embedding
```
- **Line 139**: Create an embedding vector for the search query

```python
    qvec_str = to_vec_str(qvec)
```
- **Line 140**: Convert the query vector to string format for SQL

```python
    cur = conn.cursor()
```
- **Line 142**: Create cursor for search query

```python
    cur.execute(f"""
        SELECT doc, page_num, chunk_id, content, (embedding <=> %s::vector) AS dist
        FROM {TABLE}
        ORDER BY embedding <=> %s::vector
        LIMIT {TOP_K};
    """, (qvec_str, qvec_str))
```
- **Line 143-148**: SQL query to find the TOP_K most similar chunks using vector distance (<=> operator)

```python
    rows = cur.fetchall()
```
- **Line 149**: Get all the matching rows from the database

```python
    cur.close()
```
- **Line 150**: Close the cursor

#### Display Results (Lines 152-157)
```python
    # Print what was retrieved (helps debugging)
    print(f"\nTop {TOP_K} matches for: {TEST_QUERY}")
```
- **Line 153**: Print header with query

```python
    for doc, page_num, chunk_id, content, dist in rows:
```
- **Line 154**: Loop through each matching result

```python
        print("-" * 80)
```
- **Line 155**: Print a separator line

```python
        print(f"{doc} | page {page_num} | chunk {chunk_id} | dist={dist:.4f}")
```
- **Line 156**: Print metadata about the match (document, page, chunk, distance)

```python
        print(content[:300].replace("\n", " ") + ("..." if len(content) > 300 else ""))
```
- **Line 157**: Print first 300 characters of the content (preview)

#### Generate Answer with Claude (Lines 159-181)
```python
    # Build context for Claude
    context = ""
```
- **Line 160**: Create empty string for context

```python
    for doc, page_num, chunk_id, content, dist in rows:
```
- **Line 161**: Loop through matching chunks again

```python
        context += f"\n\n[{doc} page {page_num} chunk {chunk_id} dist={dist:.4f}]\n{content}"
```
- **Line 162**: Add each chunk to the context string with metadata

```python
    prompt = (
```
- **Line 164**: Start building the prompt for Claude

```python
        "You are an insurance policy assistant.\n"
```
- **Line 165**: Tell Claude its role

```python
        "Answer the question using ONLY the context below.\n"
```
- **Line 166**: Instructions to use only provided context

```python
        "If you cannot answer from the context, say what is missing.\n\n"
```
- **Line 167**: Instructions for when context is insufficient

```python
        f"Question:\n{TEST_QUERY}\n\n"
```
- **Line 168**: Add the user's question

```python
        f"Context:\n{context}"
```
- **Line 169**: Add all the matching chunks as context

```python
    )
```
- **Line 170**: Close the prompt

```python
    # Claude answer
    msg = claude.messages.create(
```
- **Line 173**: Send the prompt to Claude

```python
        model=CLAUDE_MODEL,
```
- **Line 174**: Specify which Claude model to use

```python
        max_tokens=700,
```
- **Line 175**: Limit response to 700 tokens (words)

```python
        messages=[{"role": "user", "content": prompt}],
```
- **Line 176**: Send the prompt as a user message

```python
    )
```
- **Line 177**: Close the API call

```python
    answer = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
```
- **Line 178**: Extract the text response from Claude's answer

```python
    print("\nAnswer:\n")
```
- **Line 180**: Print header

```python
    print(answer.strip())
```
- **Line 181**: Print Claude's answer

```python
    conn.close()
```
- **Line 183**: Close the database connection

### Entry Point (Lines 186-187)
```python
if __name__ == "__main__":
```
- **Line 186**: Check if script is being run directly (not imported)

```python
    main()
```
- **Line 187**: Call the main function to start the program

---

## Pictorial Representation

### Overall Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PDF TO VECTOR PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: SETUP & CONFIGURATION
═══════════════════════════════════════
    .env file                    Configuration Variables
    ├─ DATABASE_URL      ──────► DATABASE_URL
    ├─ OPENAI_API_KEY    ──────► OPENAI_API_KEY
    ├─ OPENAI_EMBED_MODEL──────► OPENAI_EMBED_MODEL
    ├─ EMBED_DIM         ──────► EMBED_DIM
    ├─ ANTHROPIC_API_KEY ──────► ANTHROPIC_API_KEY
    ├─ CLAUDE_MODEL      ──────► CLAUDE_MODEL
    ├─ PDF_1             ──────► PDFS[0]
    ├─ PDF_2             ──────► PDFS[1]
    └─ TEST_QUERY        ──────► TEST_QUERY


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
    │  API (text-embedding)│
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


STEP 6: QUERY PROCESSING
═══════════════════════════════════════

    User Question
    "What is the coverage limit?"
         │
         ▼
    Format with prefix:
    "Insurance policy question:
     What is the coverage limit?"
         │
         ▼
    ┌──────────────────────┐
    │  OpenAI Embeddings   │
    │  (same model)        │
    └──────────────────────┘
         │
         ▼
    Query Vector
    [0.111, -0.222, 0.333, ...]  (1536 numbers)


STEP 7: VECTOR SEARCH
═══════════════════════════════════════

    Query Vector
         │
         ▼
    ┌──────────────────────────────────┐
    │  PostgreSQL Vector Search        │
    │  embedding <=> query_vector      │
    │  (calculate distances)           │
    └──────────────────────────────────┘
         │
         ▼
    Find TOP_K (12) Closest Matches
    ├─ Match 1: dist=0.1234  (Chunk from AUTO-POLICY.pdf, page 3)
    ├─ Match 2: dist=0.1456  (Chunk from AUTO-POLICY.pdf, page 2)
    ├─ Match 3: dist=0.1678  (Chunk from HOMEOWNERS-POLICY.pdf, page 5)
    └─ ... (12 total matches)


STEP 8: ANSWER GENERATION
═══════════════════════════════════════

    Top 12 Matching Chunks
         │
         ▼
    Build Context String:
    "[AUTO-POLICY.pdf page 3 chunk 0 dist=0.1234]
     Coverage limits are set at..."
    
    "[AUTO-POLICY.pdf page 2 chunk 1 dist=0.1456]
     The policy includes..."
         │
         ▼
    ┌──────────────────────┐
    │  Build Prompt        │
    │  ───────────────     │
    │  Role: Assistant     │
    │  Instructions        │
    │  Question            │
    │  Context (12 chunks) │
    └──────────────────────┘
         │
         ▼
    ┌──────────────────────┐
    │  Claude AI (Anthropic)│
    │  Generate Answer      │
    └──────────────────────┘
         │
         ▼
    Final Answer:
    "Based on the policy documents, the coverage limit
     is $500,000 for comprehensive coverage..."

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
USER QUERY                            │
    │                                 │
    ▼                                 │
[Create Query Embedding]              │
    │                                 │
    ▼                                 │
[Vector Search in DB] ────────────────┘
    │
    ▼
[Get Top 12 Matches]
    │
    ▼
[Build Context]
    │
    ▼
[Send to Claude AI]
    │
    ▼
[Generate Answer]
    │
    ▼
OUTPUT: Answer to User's Question
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

### Vector Similarity Search Concept

```
Query Vector (User's Question)
    Q = [0.11, -0.22, 0.33, 0.44, ...]

Database Vectors (Document Chunks)
    V1 = [0.12, -0.23, 0.34, 0.45, ...]  ← Close match! (distance = 0.05)
    V2 = [0.88,  0.77, -0.66, -0.55, ...] ← Far away (distance = 1.42)
    V3 = [0.15, -0.25, 0.35, 0.48, ...]  ← Close match! (distance = 0.08)
    V4 = [0.90,  0.80, -0.70, -0.60, ...] ← Far away (distance = 1.50)

    ↓ Calculate Distance (Cosine Distance)

Top K Results (ordered by similarity):
    1. V1 (distance: 0.05) ← Most similar
    2. V3 (distance: 0.08)
    3. V2 (distance: 1.42)
    4. V4 (distance: 1.50) ← Least similar
```

### Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ENVIRONMENT                              │
│  .env file with API keys, database URL, file paths, etc.        │
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
│                                              ▲                  │
│                                              │                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Query       │  │  Vector      │  │  Answer      │          │
│  │  Embedding   │─►│  Search      │─►│  Generator   │          │
│  │  (OpenAI)    │  │  (PostgreSQL)│  │  (Claude)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐  ┌──────────────┐  ┌──────────────┐
│  PDF Files      │  │  Database    │  │  External    │
│  (data/)        │  │  (PostgreSQL)│  │  APIs        │
│                 │  │              │  │  (OpenAI,    │
│  • AUTO-POLICY  │  │  • Tables    │  │   Anthropic) │
│  • HOMEOWNERS   │  │  • Vectors   │  │              │
└─────────────────┘  └──────────────┘  └──────────────┘
```

---

## Summary

This pipeline:
1. **Reads** PDF files from disk
2. **Extracts** text page by page
3. **Splits** text into smaller chunks (with overlap for context)
4. **Converts** text chunks to numerical vectors using AI (embeddings)
5. **Stores** chunks and vectors in a PostgreSQL database
6. **Searches** for similar content when given a question
7. **Retrieves** the most relevant chunks
8. **Generates** an answer using Claude AI based on the retrieved context

The key innovation is using vector embeddings to find semantically similar text, not just keyword matches!

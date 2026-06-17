from pathlib import Path

import psycopg
from litellm import embedding
from pypdf import PdfReader  # Ensure you have pypdf or PyPDF2 installed


def ingest_pdfs_to_rag():
    base_dir = Path(
        "/workspaces/Table-Reclamation-Demo/data/mathe_unstructured_dataset"
    )
    conn_info = (
        "dbname=rag user=postgres password=password host=db_rag port=5432"
    )

    # 1. Validate directory and look for PDF files
    if not base_dir.exists():
        print(f"Error: Directory not found at {base_dir}")
        return

    pdf_files = list(base_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {base_dir}")
        return

    print(f"Found {len(pdf_files)} PDFs to process.")

    # 2. Establish a single database connection for the entire batch
    with psycopg.connect(conn_info) as conn:
        print(f"Database Connection Status: {conn.info.status}")

        with conn.cursor() as cur:
            for file_path in pdf_files:
                source_document = file_path.stem  # Gets filename without .pdf
                print(f"\nProcessing: {file_path.name}...")

                try:
                    # 3. Extract text from the PDF pages
                    reader = PdfReader(file_path)
                    text = "".join(
                        [page.extract_text() or "" for page in reader.pages]
                    ).strip()

                    if not text:
                        print(
                            f"Skipping {file_path.name}: No text could be extracted."
                        )
                        continue

                    # 4. Generate LLM embedding
                    # (Assuming 'embedding' function is imported from your specific library like litellm)
                    response = embedding(
                        model="ollama/embeddinggemma:300m",
                        input=[text],
                        api_base="http://host.docker.internal:11434",
                    )

                    embeddings = response.data[0]["embedding"]

                    # 5. Execute safe parameterized insert query
                    query = """
                        INSERT INTO items (name, embedding)
                        VALUES (%s, %s::vector);                
                    """
                    cur.execute(query, (source_document, embeddings))

                    # Commit after each file to ensure progress is saved if a later file fails
                    conn.commit()
                    print(
                        f"Successfully inserted embeddings for: {source_document}")

                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
                    # Rollback the failed transaction chunk so the cursor can keep going
                    conn.rollback()


if __name__ == "__main__":
    # Assuming 'embedding' is already imported or defined in your actual script context
    ingest_pdfs_to_rag()

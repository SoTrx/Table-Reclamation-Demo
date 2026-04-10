import json
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from os import read
from pathlib import Path

import dotenv
import pytest
import tiktoken
from litellm import BaseModel, completion
from pypdf import PdfReader

from table_reclamation.facade.table_reclamation import AccessPlanner

dotenv.load_dotenv()

################# For Tiktoken tokenizer #################


def chunk_text(text, max_tokens=2000, overlap=200):
    enc = tiktoken.get_encoding("cl100k_base")  # works fine for most LLMs
    tokens = enc.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk = tokens[start:end]
        chunks.append(enc.decode(chunk))

        start += max_tokens - overlap  # overlap prevents cutting rows

    return chunks

############## PDF text extract in parallel #############


def extract_page_text(i, file_path):
    try:
        reader = PdfReader(file_path)  # reopen inside process
        return reader.pages[i].extract_text()
    except Exception as e:
        print(f"Error processing page {i}: {e}")
        return ""  # return empty string on error to avoid breaking the whole process


QUESTIONS = [
    "How can I solve a linear system 4x4?",
    "What are the basic concepts of learning linear optimization?",
    "How do I begin my study about linear optimization?",
    "What is the role of the objective function in finding the optimum solution in a linear optimization problem?",
    "What is the role of constraints in a linear optimization problem?",
    "How is the optimal solution usually found in a linear optimization problem?",
    "Fetch the assessments of student number 1273 and student number 3409",
]


@pytest.mark.parametrize("question", QUESTIONS)
def test_generate_mathe_plan(planner_mathe_split: AccessPlanner, question: str):
    planner_mathe_split.generate_stats()
    plan = planner_mathe_split.generate_plan(question)
    print(plan)
    assert len(plan) > 1


@pytest.mark.parametrize("question", QUESTIONS)
def test_generate_prompt(planner_mathe_split: AccessPlanner, question: str):
    planner_mathe_split.generate_stats()
    plan = planner_mathe_split.generate_plan(question)
    print(plan)

    class Data(BaseModel):
        # This is the column names of the sql query result.
        header: list[str]
        # This is the result of the sql query.
        data: list[list[str]]
        # explain why you got this result, and how you execute the SQL query.
        explanation: str

    class DataList(BaseModel):
        data: list[Data]

    for p in plan:
        # TODO: returned UR from Fares' model has to contain the type either 'Document' or 'Query', for now I'm hardcoding it to 'Document' to test the LLM prompt.
        p.type = 'Document'
        # added id to check for any hallucination from the model.
        p.sql = "SELECT id, student_id FROM mathe.assessment_10 WHERE student_id IN ('3409')"

        if (p.type == 'Query'):
            # TODO: Execute OP, but not my scope for now.
            pass
        elif (p.type == 'Document'):
            # Pass the document & query to LLM.
            ##### Step 1. Read Document (only PDF for now) #####
            file_path = Path(
                "/workspaces/Table-Reclamation-Demo/data/mathe_splitted/mathe.assessment_10.pdf")
            DocumentReader = PdfReader(file_path)
            text = ""
            for i in range(len(DocumentReader.pages)):
                text += DocumentReader.pages[i].extract_text()

            ##### Option) MULTIPROCESSING PDF READING #####
            # DocumentReader = PdfReader(file_path)
            # num_pages = len(DocumentReader.pages)

            # with Pool(cpu_count()) as pool:
            #     func = partial(extract_page_text, file_path=file_path)
            #     results = pool.map(func, range(num_pages))
            # text = "".join(results)

            ############ Step2. tiktoken setup ############
            chunks = chunk_text(text)
            all_results = []

            start_time = time.time()
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i}/{len(chunks)}")

                response = completion(
                    model="ollama/gemma4:e4b",
                    reasoning_effort="medium",
                    messages=[
                        {"role": "system", "content": """
                            You are a deterministic SQL execution engine.

                            Your task:
                            Execute the SQL query EXACTLY on the provided dataset.

                            --------------------------------
                            STRICT RULES (MANDATORY)
                            --------------------------------

                            1. Output MUST be valid JSON only. No text before or after.
                            2. Output MUST match EXACTLY this schema:

                            {
                            "header": ["column1", "..."],
                            "data": [["value1", "..."]],
                            "explanation": "short explanation"
                            }

                            3. DO NOT wrap output in a list.
                            4. DO NOT return multiple JSON objects.
                            5. DO NOT repeat the query.
                            6. DO NOT hallucinate values.
                            7. ONLY return rows that EXACTLY match the WHERE condition.
                            8. If NO rows match → return:
                            "data": []

                            9. Each row in "data" MUST have the SAME number of columns as "header".
                            10. NEVER return malformed rows (e.g., ["123"] if 2 columns expected).

                            --------------------------------
                            DATASET RULES
                            --------------------------------

                            - Dataset is RAW TEXT (space-separated)
                            - First line = column names
                            - Each next line = one row
                            - You MUST manually parse rows
                            - Columns are separated by spaces

                            --------------------------------
                            SQL RULES
                            --------------------------------

                            - Only use the provided dataset
                            - Apply WHERE conditions strictly
                            - SELECT only requested columns
                            - DISTINCT = remove duplicates

                            --------------------------------
                            EXAMPLES
                            --------------------------------

                            Example 1:

                            SQL:
                            SELECT id, name FROM table WHERE id IN ('36', '49');

                            Dataset:
                            id name
                            36 Jason
                            49 Alice
                            60 Bob

                            Output:
                            {
                            "header": ["id", "name"],
                            "data": [["36", "Jason"], ["49", "Alice"]],
                            "explanation": "Rows where id is 36 or 49."
                            }

                            --------------------------------

                            Example 2:

                            SQL:
                            SELECT id, name FROM table WHERE id = '10';

                            Dataset:
                            id name
                            36 Jason
                            49 Alice
                            60 Bob

                            Output:
                            {
                            "header": ["id", "name"],
                            "data": [],
                            "explanation": "No matching rows found."
                            }

                            --------------------------------

                            IMPORTANT FINAL CHECK (before answering):

                            - Is JSON valid? ✔
                            - Does each row match header length? ✔
                            - Any hallucinated values? ✘
                            - Any partial matches? ✘

                            If any rule is violated → FIX before returning.

                            --------------------------------
                            Now execute the query.
                            """},
                        {"role": "user", "content": f"""
                            DATASET:
                            {chunk}

                            SQL QUERY:
                            {p.sql}

                            Return ONLY the JSON.
                            """
                         }],
                    response_format=Data,
                    api_base="http://host.docker.internal:11434"
                )

                print(response)
                all_results.append(response)

            execution_time = time.time() - start_time

            print(all_results)
            print("Received={}".format(all_results))

            all_data = []
            headers = None

            for res in all_results:  # your list of ModelResponse
                try:
                    content = res.choices[0].message.content
                    parsed = json.loads(content)

                    # Save header once
                    if headers is None:
                        headers = parsed.get("header", [])

                    # Aggregate rows
                    all_data.extend(parsed.get("data", []))

                except Exception as e:
                    print("Skipping invalid response:", e)

            result = {
                "header": headers,
                "data": all_data
            }
            # check if the result["data"] contains two elements and the expected student_id '3409'.

            hit = 0
            for d in result["data"]:
                if len(d) >= 2 and d[1] == '3409':
                    hit += 1
            expected = 11
            accuracy = hit / expected
            hit_ratio = expected / len(result["data"])
            print(
                f'Expected: {expected}, Hit: {hit}, Accuracy(vs Expected): {accuracy:.2%}, Hit ratio = {hit_ratio:.2%}')
            print(result)


def test_litellm():
    response = completion(
        model="ollama/gemma4:e4b",
        messages=[
            {"content": "respond in 20 words. who are you?", "role": "user"}],
        api_base="http://host.docker.internal:11434"
    )
    print(response)

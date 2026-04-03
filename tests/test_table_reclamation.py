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


def chunk_text(text, max_tokens=100, overlap=200):
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
    reader = PdfReader(file_path)  # reopen inside process
    return reader.pages[i].extract_text()


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
            # Step 1. Read Document (only PDF for now)
            file_path = Path(
                "/workspaces/Table-Reclamation-Demo/data/mathe_splitted/mathe.assessment_10.pdf")
            # DocumentReader = PdfReader(file_path)
            # text = ""
            # for i in range(len(DocumentReader.pages)):
            #     text += DocumentReader.pages[i].extract_text()

            DocumentReader = PdfReader(file_path)
            num_pages = len(DocumentReader.pages)

            with Pool(cpu_count()) as pool:
                func = partial(extract_page_text, file_path=file_path)
                results = pool.map(func, range(num_pages))
            text = "".join(results)

            ############ tiktoken setup ############
            chunks = chunk_text(text)
            all_results = []
            ##########################################

            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}")
                response = completion(
                    model="ollama/qwen3.5:9b",
                    messages=[
                        {"role": "system", "content": """
                                        You are a SQL execution engine working on extracted table text.

                                        You MUST:
                                        - Execute the SQL query on the provided dataset
                                        - Return ONLY valid JSON matching the schema
                                        - NEVER repeat the query
                                        - ALWAYS extract and filter rows manually from the dataset

                                        Output schema:
                                        {
                                        "data": [
                                            {
                                            "header": ["column1", ...],
                                            "data": [["value1", ...]],
                                            "explanation": "short explanation"
                                            }
                                        ]
                                        }

                                        IMPORTANT:
                                        - Be aware there might be more than one matching row, in that case fetch all of them.
                                        - The dataset is RAW TEXT extracted from a table
                                        - Rows are space-separated
                                        - First row contains column names
                                        - You must manually parse and filter rows

                                        --------------------------------
                                        FEW-SHOT EXAMPLES
                                        --------------------------------

                                        Example 1:

                                        SQL Query:
                                        SELECT DISTINCT id, name FROM table WHERE id IN ('36', '49');

                                        Dataset:
                                        id name
                                        1 36 Jason
                                        2 49 Alice
                                        3 60 Bob

                                        Output example:
                                        {
                                        "data": [
                                            {
                                            "header": ["id", "name"],
                                            "data": [["36", "Jason"], ["49", "Alice"]],
                                            "explanation": "Filtered rows where id is in (36, 49) and returned all matching rows."
                                            }
                                        ]
                                        }

                                        --------------------------------

                                        Now execute the given query on the provided dataset.
                                        """},
                        {"role": "user", "content": f"""
                                        Return the results of the SQL query, using the dataset given below
                                        SQL Query:
                                        {p.sql}

                                        Dataset:
                                        {chunk}
                                        """
                         }],
                    # response_format=Data,
                    api_base="http://host.docker.internal:11434"
                )
                print(response)
                all_results.append(response)

            print("Received={}".format(all_results))
            data_list = DataList.model_validate_json(
                response.choices[0].message.content)
            assert (len(data_list.data) > 0)


def test_litellm():
    response = completion(
        model="ollama/llama3.1",
        messages=[
            {"content": "respond in 20 words. who are you?", "role": "user"}],
        api_base="http://host.docker.internal:11434"
    )
    print(response)

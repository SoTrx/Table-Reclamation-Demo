import json
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from os import read
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
import pytest
import tiktoken
from litellm import BaseModel, completion
from matplotlib.ticker import ScalarFormatter
from pypdf import PdfReader

from table_reclamation.facade.table_reclamation import AccessPlanner

dotenv.load_dotenv()

################# For Tiktoken tokenizer #################


def chunk_with_header(text, max_tokens=1500, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)

    lines = text.split("\n")

    header = None
    chunks = []
    current_chunk = []
    current_tokens = 0
    started = False  # ← controls when chunking begins

    for line in lines:
        if not line.strip():
            continue

        # Detect header
        if not started:
            if "student_id" in line:  # replace with your detection logic if needed
                header = line
                started = True
            continue  # ignore everything before header

        # From here: only real table rows
        row = line + "\n"
        tokens = len(enc.encode(row))

        if current_tokens + tokens > max_tokens:
            if current_chunk:
                chunk_text = header + "\n" + "".join(current_chunk)
                chunks.append(chunk_text)

            current_chunk = [row]
            current_tokens = tokens
        else:
            current_chunk.append(row)
            current_tokens += tokens

    # Final chunk
    if current_chunk:
        chunk_text = header + "\n" + "".join(current_chunk)
        chunks.append(chunk_text)

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

            # Testing with various context length (starting from 1 to doubling up to 128000) and plot the execution_time, hit_ratio, accuracy depending n the context length.
            ############ Step2. tiktoken setup ############
            context_sizes = [2**i for i in range(10, 18)]  # 1024 → 131072
            context_sizes.extend([3072, 49152, 98304])
            context_sizes.sort()
            results_log = []

            for context_size in context_sizes:
                print(f"\n=== Testing context size: {context_size} ===")

                chunks = chunk_with_header(text, max_tokens=context_size)
                all_results = []

                start_time = time.time()
                for i, chunk in enumerate(chunks):
                    print(f"Processing chunk {i}/{len(chunks)}")

                    response = completion(
                        model="ollama/gemma4:e4b",
                        # enable the model to reason about the SQL execution steps, but not too much to avoid hallucination.
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
                accuracy = hit / expected if expected else 0
                hit_ratio = hit / len(result["data"]) if result["data"] else 0

                results_log.append({
                    "context_size": context_size,
                    "execution_time": execution_time,
                    "accuracy": accuracy,
                    "hit_ratio": hit_ratio,
                    "num_chunks": len(chunks),
                    "num_rows": len(result["data"]),
                    "result": result
                })

            import matplotlib.pyplot as plt

            sizes = [r["context_size"] for r in results_log]
            times = [r["execution_time"] for r in results_log]
            accuracy = [r["accuracy"] for r in results_log]
            hit_ratio = [r["hit_ratio"] for r in results_log]

            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Primary Y-axis: Execution Time
            ax1.plot(sizes, times, color='tab:blue',
                     marker='o', label='Execution Time (s)')
            ax1.set_xlabel('Context Size')
            ax1.set_ylabel('Execution Time (s)', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.set_xscale("log")
            ax1.set_xticks(sizes)
            ax1.xaxis.set_major_formatter(ScalarFormatter())
            plt.xticks(rotation=45)

            # Secondary Y-axis for Accuracy and Hit Ratio
            ax2 = ax1.twinx()
            ax2.plot(sizes, accuracy, color='tab:green',
                     marker='s', label='Accuracy')
            ax2.plot(sizes, hit_ratio, color='tab:red',
                     marker='d', linestyle='--', label='Hit Ratio')
            ax2.set_ylabel('Accuracy / Hit Ratio')

            # Combine legends from both axes
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 +
                       labels_2, loc='upper right')

            plt.title("Execution Time, Accuracy, and Hit Ratio vs Context Size")
            plt.grid(True, which="both", ls="-", alpha=0.3)
            fig.tight_layout()
            plt.show()

            print(results_log)
            plt.savefig("metrics_plot.png")
            plt.close()


def test_litellm():
    response = completion(
        model="ollama/gemma4:e4b",
        messages=[
            {"content": "respond in 20 words. who are you?", "role": "user"}],
        api_base="http://host.docker.internal:11434"
    )
    print(response)


# results_log = [
# {'context_size': 1024, 'execution_time': 229.37085103988647, 'accuracy': 1.0, 'hit_ratio': 1.0, 'num_chunks': 63, 'num_rows': 11},
# {'context_size': 2048, 'execution_time': 166.96095371246338, 'accuracy': 1.0909090909090908, 'hit_ratio': 1.0, 'num_chunks': 31, 'num_rows': 12},
# {'context_size': 4096, 'execution_time': 127.35845589637756, 'accuracy': 0.09090909090909091, 'hit_ratio': 0.2, 'num_chunks': 16, 'num_rows': 5},
# {'context_size': 8192, 'execution_time': 67.50510740280151, 'accuracy': 0.0, 'hit_ratio': 0, 'num_chunks': 8, 'num_rows': 0},
# {'context_size': 16384, 'execution_time': 34.29264163970947, 'accuracy': 0.0, 'hit_ratio': 0.0, 'num_chunks': 4, 'num_rows': 1},
# {'context_size': 32768, 'execution_time': 16.155858993530273, 'accuracy': 0.0, 'hit_ratio': 0, 'num_chunks': 2, 'num_rows': 0},
# {'context_size': 65536, 'execution_time': 9.332051038742065, 'accuracy': 0.09090909090909091, 'hit_ratio': 1.0, 'num_chunks': 1, 'num_rows': 1},
# {'context_size': 131072, 'execution_time': 8.21578073501587, 'accuracy': 0.09090909090909091, 'hit_ratio': 1.0, 'num_chunks': 1, 'num_rows': 1}]

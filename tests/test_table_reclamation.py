import json
import os
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

# --- adjustText IMPORT WITH GRACEFUL FALLBACK ---
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    print("\n[INFO] 'adjustText' library not found. Labels in the plot may overlap.")
    print("[INFO] To fix this, run: pip install adjustText\n")
    HAS_ADJUST_TEXT = False
# ------------------------------------------------

from table_reclamation.facade.table_reclamation import AccessPlanner

dotenv.load_dotenv()
MODEL = "ollama/gemma4:31b"

################# For Tiktoken tokenizer #################


def chunk_with_header(text, max_tokens, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)

    lines = text.split("\n")

    header = None
    chunks = []
    current_chunk = []
    current_tokens = 0
    started = False  # controls when chunking begins

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
        header: list[str]
        data: list[list[str]]
        explanation: str

    class DataList(BaseModel):
        data: list[Data]

    for p in plan:
        p.type = 'Document'
        p.sql = "SELECT id, student_id FROM mathe.assessment_10 WHERE student_id IN ('3409')"

        if (p.type == 'Query'):
            pass
        elif (p.type == 'Document'):
            file_path = Path(
                "/workspaces/Table-Reclamation-Demo/data/mathe_splitted/mathe.assessment_10.pdf")
            DocumentReader = PdfReader(file_path)
            text = ""
            for i in range(len(DocumentReader.pages)):
                text += DocumentReader.pages[i].extract_text()

            context_sizes = [2**i for i in range(15, 17)]
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
                        model=MODEL,
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
                                
                                IMPORTANT FINAL CHECK (before answering):
                                - Is JSON valid? ✔
                                - Does each row match header length? ✔
                                - Any hallucinated values? ✘
                                - Any partial matches? ✘

                                If any rule is violated → FIX before returning.
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
                        api_base="http://host.docker.internal:11439",
                        timeout=3600
                    )

                    print(response)
                    all_results.append(response)

                walltime = time.time() - start_time
                print(all_results)
                print("Received={}".format(all_results))

                all_data = []
                headers = None

                for res in all_results:
                    try:
                        content = res.choices[0].message.content
                        parsed = json.loads(content)

                        if headers is None:
                            headers = parsed.get("header", [])

                        all_data.extend(parsed.get("data", []))

                    except Exception as e:
                        print("Skipping invalid response:", e)

                result = {
                    "header": headers,
                    "data": all_data
                }

                # ---------------------------------------------------------
                # CONFUSION MATRIX & METRICS LOGIC
                # ---------------------------------------------------------

                data_lines = [line for line in text.split(
                    "\n") if line.strip() and "student_id" not in line]
                total_dataset_rows = len(data_lines)

                actual_p = 11  # Known expected positives for student_id '3409'
                actual_n = max(0, total_dataset_rows - actual_p)

                tp = 0
                fp = 0

                for d in result["data"]:
                    if len(d) >= 2 and d[1] == '3409':
                        tp += 1
                    else:
                        fp += 1

                fn = max(0, actual_p - tp)
                tn = max(0, actual_n - fp)

                # Advanced Metrics Calculations (with zero-division protection)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                fall_out = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                f1_denominator = precision + recall
                f1_score = 2 * (precision * recall) / \
                    f1_denominator if f1_denominator > 0 else 0.0

                total_population = tp + tn + fp + fn
                accuracy = (tp + tn) / \
                    total_population if total_population > 0 else 0.0

                results_log.append({
                    "context_size": context_size,
                    "walltime": walltime,
                    "TP": tp,
                    "TN": tn,
                    "FP": fp,
                    "FN": fn,
                    "Recall": recall,
                    "Precision": precision,
                    "Specificity": specificity,
                    "Fall-out": fall_out,
                    "F1-Score": f1_score,
                    "Accuracy": accuracy,
                    "num_chunks": len(chunks),
                    "num_rows": len(result["data"]),
                    "result": result
                })

            # --- JSON EXPORT CODE ---
            log_file_path = "logs.json"
            existing_logs = {}

            if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
                try:
                    with open(log_file_path, "r", encoding="utf-8") as f:
                        existing_logs = json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"Warning: '{log_file_path}' contains invalid JSON. Starting fresh.")

            if MODEL in existing_logs:
                existing_logs[MODEL].extend(results_log)
            else:
                existing_logs[MODEL] = results_log

            with open(log_file_path, "w", encoding="utf-8") as f:
                json.dump(existing_logs, f, indent=4)

            # --- PRECISION-RECALL PLOT CODE ---
            import matplotlib.pyplot as plt

            recall_list = [r["Recall"] for r in results_log]
            precision_list = [r["Precision"] for r in results_log]
            sizes = [r["context_size"] for r in results_log]

            fig, ax = plt.subplots(figsize=(10, 8))

            # 1. Plot the main trajectory line and points
            ax.plot(
                recall_list,
                precision_list,
                marker='o',
                markersize=8,
                color='tab:purple',
                linestyle='-',
                label='Model Performance Trajectory'
            )

            # 2. Collect all text annotation objects
            texts = []
            for i, txt in enumerate(sizes):
                texts.append(ax.annotate(
                    f"{txt}",
                    (recall_list[i], precision_list[i]),
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="gray", alpha=0.8)
                ))

            # 3. Perform the automatic label adjustment
            if HAS_ADJUST_TEXT:
                print(
                    "[INFO] Adjusting labels to prevent overlap... this may take a moment.")
                adjust_text(
                    texts,
                    x=recall_list,
                    y=precision_list,
                    expand_points=(2.0, 2.0),
                    force_text=(0.3, 0.6),
                    force_points=(0.2, 0.5),
                    lim=500,
                    arrowprops=dict(arrowstyle="->",
                                    color='gray', lw=0.5, alpha=0.7)
                )
            else:
                # Fallback if adjustText is not installed
                for t in texts:
                    t.set_va('center')
                    t.set_ha('center')

            ax.set_title("Precision-Recall Space by Context Size")
            ax.set_xlabel("Recall (True Positive Rate)")
            ax.set_ylabel("Precision (Positive Predictive Value)")

            # Padding
            ax.set_xlim([-0.08, 1.08])
            ax.set_ylim([-0.08, 1.08])

            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.legend(loc='upper right')

            fig.tight_layout()

            # Save and show
            plt.savefig(f"pr_plot_{MODEL[7:]}.png")
            plt.show()
            plt.close()


def test_litellm():
    response = completion(
        model=MODEL,
        messages=[
            {"content": "respond in 20 words. who are you?", "role": "user"}],
        api_base="http://host.docker.internal:11439"
    )
    print(response)

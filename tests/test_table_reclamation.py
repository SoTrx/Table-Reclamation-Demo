from os import read
from pathlib import Path

import dotenv
import pytest
from litellm import BaseModel, completion
from pypdf import PdfReader

from table_reclamation.facade.table_reclamation import AccessPlanner

dotenv.load_dotenv()


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
        p.type = 'Document'
        p.sql = "SELECT id, student_id FROM mathe.assessment_10 WHERE student_id IN ('3409')"
        # p = {SqlOperation(src_idx=1,
        #                   table='mathe.assessment_10',
        #                   sql="SELECT DISTINCT student_id FROM mathe.assessment_10 WHERE student_id IN ('3409')",
        #                   type='Query'),

        #      plan = [SqlOperation(src_idx=1,
        #              table='mathe.assessment_10',
        #              sql="SELECT DISTINCT student_id ...ERE student_id IN ('3409')",
        #              type='Query'),
        #              SqlOperation(src_idx=5,
        #              table='mathe.assessment_4',
        #              sql="SELECT DISTINCT student_id F...ERE student_id IN ('1273')",
        #              type='Query')]}
        if (p.type == 'Query'):
            # TODO: Execute OP, but not my scope for now.
            pass
        elif (p.type == 'Document'):

            # Pass the document & query to LLM.
            # Step 1. Read Document (only PDF for now)
            file_path = Path(
                "/workspaces/Table-Reclamation-Demo/data/mathe_splitted/mathe.assessment_10.pdf")
            DocumentReader = PdfReader(file_path)
            # text = DocumentReader.pages[0].extract_text()
            text = 'id student_id question_id topic subtopic question_level answer date\n20984 3412 297 17 1 1 2024-06-11 01:14:54+00\n20985 3416 139 18 7 2 0 2024-06-11 01:15:04+00\n20986 3415 648 18 7 2 0 2024-06-11 01:15:13+00\n20987 3416 633 18 7 1 1 2024-06-11 01:15:20+00\n20988 3409 135 18 7 1 0 2024-06-11 01:15:21+00\n'
            # text = ""
            # for i in range(len(DocumentReader.pages)):
            #     text += DocumentReader.pages[i].extract_text()

            response = completion(

                model="ollama/qwen3.5:9b",
                temperature=0,
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
                                    {text}
                                    """
                     }],
                # response_format=Data,
                api_base="http://host.docker.internal:11434"
            )
            print(response)
            print("Received={}".format(response))
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

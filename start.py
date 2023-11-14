from llama_index import PromptTemplate
from llama_index.program import LLMTextCompletionProgram
from llama_index.output_parsers import PydanticOutputParser
from llama_index.llms import Bedrock
from pydantic import BaseModel, Field
from typing import List
import json


# Create the Pydantic classes to model the business problem
class Step(BaseModel):
    step_id: int
    description: str
    rationale: str
    expected_outcomes: str
    metrics_for_success: str


class ActionPlan(BaseModel):
    steps: List[Step]


class Solution(BaseModel):
    solution_id: int
    description: str
    solution_keywords: List[str] = Field(
        ..., description="List of keywords that relate to the solution description")
    rationale: str
    action_plan: ActionPlan


class BusinessProblemModel(BaseModel):
    business_problem: str
    query_keywords: List[str] = Field(
        ..., description="List of keywords that relate to the business problem")
    solutions: List[Solution] = Field(...)
    business_type: str = Field(..., description="Type of business")
    business_impact: str = Field(..., description="Impact of business")
    user_objective: str = Field(..., description="User objective")

# Let's modify the query to pass in the labels for the particular data
# Load the business-questions-v3.json
# Let's loop through the data and run inference to generate synthetic data


def generate_data(program):
    file = open('data/bq-text.json', 'r')
    data = json.load(file)
    syn_data = {}
    for index, item in enumerate(data):
        query = item['Question']
        labels = item['Labels']
        try:
            output = program(business_problem=query, labels=f"{labels}")
            syn_data[index] = {"original_question": item['Question'],
                               "synthetic_response": output.model_dump()}
        except Exception as e:
            print(e)
            continue
    return syn_data

# Write syn_data to file
def write_synthetic(syn_data):
    file = open("./data/output/syn_data_v2.txt", "w", encoding="utf-8")
    for example in syn_data:
        file.write(json.dumps(syn_data[example]))
        file.write("\n")


template_1 = PromptTemplate(
    "Given the following business problem, please suggest a set of solutions with detailed action plans. The business problem is {business_problem}.")
template_2 = PromptTemplate(
    "Given the following business problem and labels, please suggest a set of solutions with detailed action plans. The business problem is {business_problem}. The labels are {labels}")
llm = Bedrock(model="anthropic.claude-instant-v1",
              max_tokens=8000, max_retries=2)


program = LLMTextCompletionProgram(
    llm=llm,
    output_parser=PydanticOutputParser(BusinessProblemModel),
    verbose=True,
    prompt=template_2
)

syn_data = generate_data(program)
print(syn_data)
#write_synthetic()

#file = open("./data/output/syn_data_v2.txt", "r", encoding="utf-8")
#data = file.readlines()
#json_list = [json.loads(line) for line in data]
#print(json_list)

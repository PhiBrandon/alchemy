import os
from llama_index import PromptTemplate
from llama_index.program import LLMTextCompletionProgram
from llama_index.output_parsers import PydanticOutputParser
from llama_index.llms import Bedrock
import json
from llama_index.llms import OpenAI
from llama_index import PromptTemplate
from business_classes import BusinessProblemModel
from dotenv import load_dotenv
load_dotenv()


# Read openai key from env file

# Read openai key from env file
openai_key = os.getenv("OPENAI_KEY")


# Function to generate data
def generate_data(program, template):
    with open('data/bq-text.json', 'r') as file:
        data = json.load(file)

    syn_data = {}
    for index, item in enumerate(data):
        business_problem = item["Question"]
        business_type = item["Labels"]["Business Type"]
        business_impact = item["Labels"]["Business Impact"]
        user_objective = item["Labels"]["User Objective"]
        prompt = template.format(business_problem=business_problem,

                                 business_type=business_type,

                                 business_impact=business_impact,

                                 user_objective=user_objective,)
        try:
            output = program(prompt_template_str=prompt)
            syn_data[index] = {"original_question": item['Question'],
                               "synthetic_response": output.model_dump()}
        except Exception as e:
            print(f"Error processing item {index}: {e}")
            continue

    return syn_data

# Function to write synthetic data to file


def write_synthetic(syn_data):
    with open("./data/output/synthetic_gpt_35_test.json", "w", encoding="utf-8") as file:
        for example in syn_data.values():
            file.write(json.dumps(example) + "\n")


# Prompt templates and LLM program initialization
template_1 = PromptTemplate(
    "Given the following business problem, please suggest a set of solutions with detailed action plans. The business problem is {business_problem}.")
template_2 = PromptTemplate(
    "Given the following business problem and labels, please suggest a set of solutions with detailed action plans. The business problem is {business_problem}. The labels are {labels}")

template_3 = PromptTemplate(
    """Given a business problem in the {business_type} sector about {business_impact},

where the objective is {user_objective}, provide a detailed action plan.

The business problem is: '{business_problem}'.

Please include specific steps and rationale for each step."""
)
# llm = Bedrock(model="anthropic.claude-instant-v1", max_tokens=8000, max_retries=2)
llm = OpenAI(model="gpt-3.5-turbo-1106",
             api_key=openai_key)

program = LLMTextCompletionProgram.from_defaults(
    llm=llm,
    output_parser=PydanticOutputParser(BusinessProblemModel),
    verbose=True,
    prompt_template_str=""

)

# program = make_program()
# Data generation and writing to file
syn_data = generate_data(program, template_3)
write_synthetic(syn_data)

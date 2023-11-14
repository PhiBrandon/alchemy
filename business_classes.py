from pydantic import BaseModel, Field
from typing import List

# Pydantic classes for data modeling
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
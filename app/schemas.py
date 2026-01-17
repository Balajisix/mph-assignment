from pydantic import BaseModel, Field
from typing import List

class ResearchResponse(BaseModel):
    summary: str = Field(..., description="A concise summary of the research findings.")
    key_facts: List[str] = Field(..., description="A list of 3-5 key facts derived from the research.")
    sources: List[str] = Field(default_factory=list, description="A list of URLs or references used during the research.")

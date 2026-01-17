from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.samp import perform_research, get_mermaid_graph
from app.schemas import ResearchResponse
from pydantic import BaseModel

class ResearchRequest(BaseModel):
    query: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST", "GET", "*"],
    allow_headers=["*"],
)

@app.get('/')
def health_check():
    return {"status": "ok"}

@app.post("/api/research", response_model=ResearchResponse)
def get_research(payload: ResearchRequest):
    query = payload.query
    return perform_research(query)

@app.get("/api/graph")
def get_graph():
    return {"mermaid": get_mermaid_graph()}

import operator
from typing import Annotated, Literal, TypedDict, Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.tools import tools
from app.schemas import ResearchResponse
from app.llm import get_llm


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

llm = get_llm()
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

def call_agent(state: AgentState):
    """The reasoning node that decides whether to use tools."""
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def generate_final_report(state: AgentState):
    """The formatting node that creates the final Pydantic-validated JSON."""
    last_info = state["messages"][-1].content
    format_prompt = f"Format this research into JSON:\n\n{last_info}\n\n{parser.get_format_instructions()}"
    final_output = llm.invoke([HumanMessage(content=format_prompt)])
    return {"messages": [final_output]}

def should_continue(state: AgentState) -> Literal["tools", "finalize"]:
    last_message = state['messages'][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "finalize"

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_agent)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("finalize", generate_final_report)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "finalize": "finalize"
    }
)

workflow.add_edge("tools", "agent")
workflow.add_edge("finalize", END)

app_graph = workflow.compile()


def get_mermaid_graph():
    """Returns the Mermaid string for frontend visualization."""
    return app_graph.get_graph().draw_mermaid()

def perform_research(topic: str):
    """Executes the graph and returns the parsed JSON response."""
    initial_state = {"messages": [HumanMessage(content=f"Research: {topic}")]}
    result = app_graph.invoke(initial_state)
    return parser.parse(result["messages"][-1].content)

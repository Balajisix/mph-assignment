from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.tools import tools
from app.schemas import ResearchResponse
from app.llm import get_llm

llm = get_llm()

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

research_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional research assistant. Use tools to find details."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, research_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

def perform_research(topic: str):
    agent_response = agent_executor.invoke({"input": f"Research this topic: {topic}"})
    raw_info = agent_response["output"]
    
    format_prompt = f"""
    Turn the following research information into a valid JSON object.
    
    Research Info: {raw_info}
    
    {parser.get_format_instructions()}
    """
    
    final_output = llm.invoke(format_prompt)
    
    return parser.parse(final_output.content)

from __future__ import annotations
import ast
import operator as op
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper

_ALLOWED_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
    ast.USub: op.neg,
}

def _eval_expr(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _ALLOWED_OPS[type(node.op)](_eval_expr(node.left), _eval_expr(node.right))
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPS[type(node.op)](_eval_expr(node.operand))
    raise ValueError("Expression not allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression safely. Example: '2 + 2' or '10 * 5'."""
    try:
        parsed = ast.parse(expression, mode="eval")
        return str(_eval_expr(parsed.body))
    except Exception as e:
        return f"Error: {e}"

@tool
def now_ist() -> str:
    """Get the current time in IST (Asia/Kolkata) in ISO format."""
    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()

@tool
def get_latest_news(query: str) -> str:
    """Search for the latest news headlines about a specific topic."""
    wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", max_results=5)
    search = DuckDuckGoSearchRun(api_wrapper=wrapper)
    return search.run(f"latest news today: {query}")

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for factual and historical information."""
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1))
    return wiki.run(query)

tools = [calculator, now_ist, get_latest_news, wikipedia_search]
"""
RAG Agent using LangGraph's create_agent.
"""

# from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent

from src.config import settings
from langchain_openai import ChatOpenAI



def get_llm() -> ChatOpenAI:
    """Initialize the LLM."""
    return ChatOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model=settings.LLM_MODEL,
        temperature=temp,
        max_tokens=None,
        timeout=None,
        max_retries=settings.LLM_MAX_RETRIES,
    )


def create_rag_agent(tools: list, system_prompt: str | None = None, **kwargs):
    """
    Create a RAG agent with the given tools.

    Args:
        tools: List of tools (e.g., retrieval tool)
        system_prompt: Custom system prompt. If None, uses default.

    Returns:
        LangGraph agent
    """
    llm = get_llm(temp=kwargs.get("temp"))

    prompt = system_prompt or (
        "You have access to a tool that retrieves context from documents. "
        "Use the tool to help answer user queries."
    )
    
    return create_react_agent(llm, tools, prompt=prompt)


def run_agent(agent, query: str, stream: bool = True):
    """
    Run the agent with a query.

    Args:
        agent: The agent to run
        query: User query
        stream: Whether to stream output (default True)

    Returns:
        Final response if not streaming
    """
    messages = [{"role": "user", "content": query}]

    if stream:
        for event in agent.stream(
            {"messages": messages},
            stream_mode="values",
        ):
            event["messages"][-1].pretty_print()
        return None
    else:
        result = agent.invoke({"messages": messages})
        return result["messages"][-1].content

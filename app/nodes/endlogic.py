from typing import Literal
from langgraph.graph import StateGraph, START, END

def should_continue(state: MessageState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    mensagens = state["mensagens"]
    last_mensagem = mensagens[-1]
    
    if last_mensagem.tool.calls:
        return "tool_node"
    return END
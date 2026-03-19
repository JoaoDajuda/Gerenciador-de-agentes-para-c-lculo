import os
import operator
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage

# 1. Configuração de Ambiente
load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 2. Definição das Ferramentas
@tool 
def multiplique(a: int, b: int) -> int:
    """Multiplica a e b."""
    return a * b

@tool 
def add(a: int, b: int) -> int:
    """Soma a e b."""
    return a + b

@tool 
def divida(a: int, b: int) -> float:
    """Divide a por b."""
    return a / b

tools = [add, multiplique, divida]
tools_by_name = {tool.name: tool for tool in tools}
model_withtools = model.bind_tools(tools)

# 3. Definição do Estado do Grafo
class MensagensState(TypedDict):
    # O operator.add faz com que novas mensagens sejam anexadas à lista
    mensagens: Annotated[list, operator.add]

# 4. FUNÇÕES DOS NÓS (Devem vir ANTES do workflow)
def call_model(state: MensagensState):
    """Nó que chama o LLM"""
    resposta = model_withtools.invoke(state["mensagens"])
    return {"mensagens": [resposta]}

def tool_node(state: MensagensState):
    """Nó que executa as ferramentas se o LLM pedir"""
    last_msg = state["mensagens"][-1]
    results = []
    for tool_call in last_msg.tool_calls:
        tool_func = tools_by_name[tool_call["name"]]
        saida = tool_func.invoke(tool_call["args"])
        results.append(ToolMessage(content=str(saida), tool_call_id=tool_call["id"]))
    return {"mensagens": results}

def should_continue(state: MensagensState):
    """Lógica condicional para saber se vai para ferramentas ou para o fim"""
    last_msg = state["mensagens"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END

# 5. Montagem do Grafo (Agora as funções acima já existem para o Python)
workflow = StateGraph(MensagensState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# Variável que o LangGraph busca para rodar o servidor
app = workflow.compile()
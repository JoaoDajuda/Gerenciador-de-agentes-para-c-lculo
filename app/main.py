from langchain.tools import tool
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "claude-sonnet-4-6",
    temperature = 0
)

@tool 
def multiplique(a: int, b: int ) -> int:
    """Multiply `a` and `b`.
    
    Args:
        a: First int
        b: Second int
        
    """
    return a*b
    
@tool 
def add(a: int, b: int ) -> int:
    """Multiply `a` and `b`.
    
    Args:
        a: First int
        b: Second int
        """
    return a+b
    
@tool 
def divida(a: int, b: int ) -> float:
    """Multiply `a` and `b`.
    
    Args:
        a: First int
        b: Second int
        """
    return a/b
    
tools = [add, multiplique, divida ]
tools_by_name = {tool.name: tool for tool in tools}
model_withtools = model.bind_tools(tools)
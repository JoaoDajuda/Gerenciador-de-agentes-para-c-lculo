from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

class MensagensState(TypedDict):
    mensagens: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

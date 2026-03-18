from langchain.messages import SystemMessage

def llm_call(state: dict):
    """LLM decides wheter to call a tool or not"""
    
    return {
        "mensagens": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state ["mensagens"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }
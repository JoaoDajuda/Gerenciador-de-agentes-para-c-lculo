agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")


agent = agent_builder.compile()

from IPython.display import Image, display
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

from langchain.mensagens import HumanMessage
mensagens = [HumanMessage(content="Add 3 and 4.")]
mensagens = agent.invoke({"mensagens": mensagens})
for m in mensagens["mensagens"]:
    m.pretty_print()
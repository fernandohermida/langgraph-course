from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END, MessagesState, StateGraph

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {END: END, ACT: ACT})

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


# Same graph, different paths: 0 rounds, 1 round, 4 chained rounds, 1 parallel round.
PROMPTS = [
    "What are your support hours?",
    "Has the shipment with tracking number T-7 been delivered?",
    "I'm jane@x.com, my last order never arrived. Can I get a refund?",
    "Compare the status of shipments T-9 and T-7.",
]


def trace(prompt: str) -> None:
    """Stream the graph and print what each node adds to the state."""
    print(f"\n=== {prompt}")
    rounds = 0
    for update in app.stream(
        {"messages": [HumanMessage(content=prompt)]}, stream_mode="updates"
    ):
        for node, output in update.items():
            for message in output["messages"]:
                if isinstance(message, ToolMessage):
                    print(f"{node:<12} -> {message.name}: {message.content}")
                elif message.tool_calls:
                    rounds += 1
                    calls = ", ".join(
                        f"{c['name']}({c['args']})" for c in message.tool_calls
                    )
                    print(f"{node:<12} -> tool_calls: {calls}")
                else:
                    print(f"{node:<12} -> answer: {message.content}")
    print(f"--- {rounds} tool round(s)")


if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    for prompt in PROMPTS:
        trace(prompt)

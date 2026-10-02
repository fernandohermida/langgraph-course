"""
Lesson 2: human-in-the-loop refunds.

Same support graph as main.py, plus a human_review node in front of every refund.
- interrupt(payload) pauses the graph and hands `payload` to the caller.
- A checkpointer saves the paused state, keyed by thread_id, so the run can continue later.
- Command(resume=value) continues the same thread; `value` becomes interrupt()'s return value.
- A node can return Command(goto=...) to choose the next node itself.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from main import ACT, AGENT_REASON, LAST
from nodes import run_agent_reasoning, tool_node

HUMAN_REVIEW = "human_review"


def route(
    state: MessagesState,
) -> Literal["human_review", "run_support_tools", "__end__"]:
    tool_calls = state["messages"][LAST].tool_calls
    if not tool_calls:
        return END
    if any(call["name"] == "issue_refund" for call in tool_calls):
        return HUMAN_REVIEW
    return ACT


def human_review(
    state: MessagesState,
) -> Command[Literal["run_support_tools", "support_agent"]]:
    tool_calls = state["messages"][LAST].tool_calls
    refunds = [
        call["args"]["order_id"]
        for call in tool_calls
        if call["name"] == "issue_refund"
    ]
    approved = interrupt({"order_ids": refunds, "question": "Approve refund?"})
    if approved:
        return Command(goto=ACT)
    # Declined: answer every pending tool call ourselves and let the agent reply.
    declined = [
        ToolMessage(content="Declined by a human reviewer", tool_call_id=call["id"])
        for call in tool_calls
    ]
    return Command(goto=AGENT_REASON, update={"messages": declined})


flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.add_node(HUMAN_REVIEW, human_review)
flow.add_node(ACT, tool_node)
flow.add_edge(START, AGENT_REASON)
flow.add_conditional_edges(AGENT_REASON, route)
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    app.get_graph().draw_mermaid_png(output_file_path="hitl.png")

    config = {"configurable": {"thread_id": "1"}}
    prompt = "I'm jane@x.com, my last order never arrived. Can I get a refund?"
    print(f"=== {prompt}")

    result = app.invoke({"messages": [HumanMessage(content=prompt)]}, config)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        answer = input(
            f"{payload['question']} {', '.join(payload['order_ids'])} (y/n): "
        )
        result = app.invoke(Command(resume=answer.strip().lower() == "y"), config)

    print(f"answer: {result['messages'][LAST].content}")

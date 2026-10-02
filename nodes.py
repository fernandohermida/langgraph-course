from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

SYSTEM_MESSAGE = """
You are a customer-support agent for an online shop.
Use the tools to look up customers, orders and shipments, and to issue refunds.
Only issue a refund when the tools show the customer is entitled to one.
Support hours are Monday to Friday, 9:00 to 17:00.
"""


def run_agent_reasoning(state: MessagesState) -> dict:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]
    )
    return {"messages": [response]}


tool_node = ToolNode(tools)

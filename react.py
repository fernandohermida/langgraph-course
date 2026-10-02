from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

# Fake support-desk data: deterministic, so every run gives the same answers.
CUSTOMERS = {"jane@x.com": "C-101"}

ORDERS = {
    "C-101": [
        {"order_id": "O-555", "total_eur": 42.0, "tracking_no": "T-9", "placed": "2026-09-20"},
        {"order_id": "O-480", "total_eur": 18.5, "tracking_no": "T-7", "placed": "2026-08-02"},
    ]
}

SHIPMENTS = {"T-9": "lost in transit", "T-7": "delivered"}

REFUNDABLE_STATUSES = {"lost in transit", "damaged"}


@tool
def find_customer(email: str) -> str:
    """
    param email: the customer's email address
    returns: the customer id, or a not-found message
    """
    return CUSTOMERS.get(email.lower(), f"No customer found for {email}")


@tool
def get_orders(customer_id: str) -> list[dict] | str:
    """
    param customer_id: a customer id such as C-101
    returns: the customer's orders (id, total, tracking number, date placed)
    """
    return ORDERS.get(customer_id, f"No orders found for {customer_id}")


@tool
def get_shipment(tracking_no: str) -> str:
    """
    param tracking_no: a shipment tracking number such as T-9
    returns: the current shipment status
    """
    return SHIPMENTS.get(tracking_no, f"Unknown tracking number {tracking_no}")


@tool
def issue_refund(order_id: str) -> str:
    """
    param order_id: the order to refund, such as O-555
    returns: confirmation, or the reason the refund was refused.
    Refunds are only allowed when the shipment was lost or damaged.
    """
    for orders in ORDERS.values():
        for order in orders:
            if order["order_id"] == order_id:
                status = SHIPMENTS.get(order["tracking_no"])
                if status in REFUNDABLE_STATUSES:
                    return f"Refunded €{order['total_eur']:.2f} for {order_id}"
                return f"Refund refused for {order_id}: shipment status is '{status}'"
    return f"Unknown order {order_id}"


tools = [find_customer, get_orders, get_shipment, issue_refund]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

import os
from typing import Dict, Any, List, TypedDict, Annotated, Sequence, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from database import DatabaseManager
from tools import get_inventory_levels, get_shipping_status, weather_impact_analysis

class ItemRequest(TypedDict):
    sku: str
    quantity: int
    location: Optional[str]

# State definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Structured requests
    requested_items: List[ItemRequest]
    # Data results
    inventory_data: Dict[str, Any]
    shipping_data: Dict[str, Any]
    weather_data: Dict[str, Any]
    catalog_results: List[Dict[str, Any]]
    demand_history: List[Dict[str, Any]]
    # Status
    validation_status: Optional[str]

# Use a mock API key for build-time validation if not present
api_key = os.getenv("GROQ_API_KEY", "gsk_mock_key_for_build")

# LLM initializations
brain_llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)
support_llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key)

db_manager = DatabaseManager()

# --- PROMPTS ---
ORCHESTRATOR_PROMPT = """You are the Lead Supply Chain Orchestrator. Your job is to analyze the user request and decide which specialized agents are needed.
Available agents:
- Sales: Processes natural language orders, extracts SKUs and quantities, checks demand history.
- Warehouse: Checks real-time stock and inventory levels.
- Logistics: Manages routes, shipping status, and weather risks.

Current request: {context}

Return a comma-separated list of agents that should process this request (e.g., 'Sales, Warehouse, Logistics') or 'End' if no action is needed.
"""

SALES_PROMPT = """You are the Senior Sales Agent. Extract SKUs and quantities from the order. Analyze demand history.
Order: {context}
Provide a JSON-like summary of the order items and demand context.
"""

WAREHOUSE_PROMPT = """You are the Warehouse Agent. Report on stock levels for the requested items.
Items: {context}
"""

LOGISTICS_PROMPT = """You are the Logistics Agent. Provide shipping status and weather risk assessment.
Context: {context}
"""

REFLECTION_PROMPT = """You are the Senior Logistics Validator.
CRITICAL RULES:
1. REJECT if weather risk factor > 0.7.
2. REJECT if stock for any item is < 10% of the requested quantity.
Current context: {solution}
Output 'VALID' or 'REJECTED: [detailed reasons]'.
"""

# --- AGENT FUNCTIONS ---

async def sales_agent(state: AgentState):
    last_message = state['messages'][-1].content
    # Simulation: Parse natural language (in a real scenario, use LLM to extract JSON)
    # For now, we mock the extraction based on the search
    catalog = await db_manager.query_catalog(last_message)
    requested = []
    demand = []
    if catalog:
        # Mocking extraction of 100 units for the first found item
        sku = catalog[0]['sku']
        requested.append({"sku": sku, "quantity": 100, "location": "New York"})
        demand = await db_manager.get_demand_history(sku)

    prompt = SALES_PROMPT.format(context=last_message)
    response = await support_llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content=f"Catalog: {catalog}")])

    return {
        "messages": [response],
        "catalog_results": catalog,
        "requested_items": requested,
        "demand_history": demand
    }

async def warehouse_agent(state: AgentState):
    inventory = {}
    items = state.get('requested_items', [])
    for item in items:
        sku = item['sku']
        inventory[sku] = await get_inventory_levels(sku)

    prompt = WAREHOUSE_PROMPT.format(context=str(items))
    response = await support_llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content=f"Inventory: {inventory}")])

    return {
        "messages": [response],
        "inventory_data": inventory
    }

async def logistics_agent(state: AgentState):
    weather = {}
    shipping = {}
    items = state.get('requested_items', [])

    # Use location from first item if available
    location = items[0].get('location', "Global Hub") if items else "Global Hub"

    weather = await weather_impact_analysis(location)
    shipping = await get_shipping_status("ORD-DEFAULT")

    prompt = LOGISTICS_PROMPT.format(context=f"Location: {location}")
    response = await support_llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content=f"Weather: {weather}, Shipping: {shipping}")])

    return {
        "messages": [response],
        "weather_data": weather,
        "shipping_data": shipping
    }

async def validation_node(state: AgentState):
    weather_risk = state.get('weather_data', {}).get('risk_factor', 0)
    inventory = state.get('inventory_data', {})
    requested_items = state.get('requested_items', [])

    reasons = []
    if weather_risk > 0.7:
        reasons.append(f"Weather risk too high ({weather_risk} > 0.7)")

    for req in requested_items:
        sku = req['sku']
        qty_requested = req['quantity']
        qty_available = inventory.get(sku, {}).get('quantity', 0)

        if qty_available < (0.1 * qty_requested):
            reasons.append(f"Critical stock for {sku}: requested {qty_requested}, only {qty_available} available (less than 10%)")

    status = "VALID" if not reasons else f"REJECTED: {'; '.join(reasons)}"

    solution_summary = f"Inventory: {inventory}, Weather: {weather_risk}, Requested: {requested_items}, Status: {status}"
    prompt = REFLECTION_PROMPT.format(solution=solution_summary)
    response = await brain_llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content="Final validation review.")])

    return {
        "messages": [response],
        "validation_status": status
    }

async def orchestrator_router(state: AgentState):
    last_message = state['messages'][-1].content
    prompt = ORCHESTRATOR_PROMPT.format(context=last_message)
    response = await brain_llm.ainvoke([SystemMessage(content=prompt)])

    decision = response.content.lower()
    if "end" in decision:
        return "End"

    selected_agents = []
    if "sales" in decision: selected_agents.append("Sales")
    if "warehouse" in decision: selected_agents.append("Warehouse")
    if "logistics" in decision: selected_agents.append("Logistics")

    # If no agents selected but not 'End', default to Sales for safety
    if not selected_agents:
        return "Sales"

    # Return list for parallel execution if multiple
    return selected_agents

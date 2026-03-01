import os
import json
import re
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

CRITICAL: If the request is complex and involves multiple aspects (e.g., stock AND weather), you MUST return all relevant agents separated by commas.
Return a comma-separated list of agents that should process this request (e.g., 'Sales, Warehouse, Logistics') or 'End' if no action is needed.
"""

SALES_PROMPT = """You are the Senior Sales Agent.
1. Identify the products and quantities requested.
2. If the user mentions a product naturally (e.g., 'chips'), use the catalog data to find the SKU.
3. Extract any location mentioned for delivery.
Order: {context}
Catalog Information: {catalog}
Demand History: {demand}

Provide a summary of the order including SKUs, quantities, and a justification based on demand history if the quantity seems reasonable or unusual.
"""

WAREHOUSE_PROMPT = """You are the Warehouse Agent.
Report on the current inventory levels for the following requested items: {context}
Inventory Data: {inventory}
State clearly if the requested quantities are available or if there is a shortage.
"""

LOGISTICS_PROMPT = """You are the Logistics Agent.
Analyze the following logistics context: {context}
Weather Data: {weather}
Shipping Data: {shipping}

Provide an assessment of the weather risk and the current status of any orders mentioned.
"""

REFLECTION_PROMPT = """You are the Senior Logistics Validator.
Analyze the current situation:
{solution}

CRITICAL DECISION RULES:
1. If weather risk factor > 0.7, you MUST propose an alternative (e.g., 'wait 2 days', 'reroute through different hub') instead of a simple rejection.
2. If stock for any item is < 10% of requested quantity, you MUST propose an alternative (e.g., 'partial shipment', 'ship from alternative warehouse B', 'wait for restock').
3. If neither rule is triggered, the status is 'VALID'.

Output your decision in this format:
STATUS: [VALID | PROPOSAL]
REASON: [Reason for proposal or validation]
ALTERNATIVE: [Description of the alternative if STATUS is PROPOSAL]
"""

# --- AGENT FUNCTIONS ---

async def sales_agent(state: AgentState):
    last_message = state['messages'][-1].content

    # 1. Query catalog to identify SKUs
    catalog = await db_manager.query_catalog(last_message)

    # Use LLM to extract structured information
    extraction_prompt = f"Extract SKUs, quantities and location from this message: '{last_message}'. Use this catalog for SKUs: {catalog}. Return a JSON list of objects with 'sku', 'quantity', and 'location'."
    extraction_response = await support_llm.ainvoke([SystemMessage(content="You are a data extraction assistant. Return ONLY valid JSON."), HumanMessage(content=extraction_prompt)])

    try:
        # Simple extraction logic
        json_match = re.search(r'\[.*\]', extraction_response.content, re.DOTALL)
        if json_match:
            requested = json.loads(json_match.group())
        else:
            requested = []
    except:
        requested = []

    # 2. Get demand history for each SKU to justify quantities
    demand_history = {}
    for item in requested:
        sku = item['sku']
        demand_history[sku] = await db_manager.get_demand_history(sku)

    prompt = SALES_PROMPT.format(context=last_message, catalog=catalog, demand=demand_history)
    response = await support_llm.ainvoke([SystemMessage(content=prompt)])

    return {
        "messages": [response],
        "catalog_results": catalog,
        "requested_items": requested,
        "demand_history": [demand_history]
    }

async def warehouse_agent(state: AgentState):
    inventory = {}
    items = state.get('requested_items', [])
    for item in items:
        sku = item['sku']
        inventory[sku] = await get_inventory_levels(sku)

    prompt = WAREHOUSE_PROMPT.format(context=str(items), inventory=inventory)
    response = await support_llm.ainvoke([SystemMessage(content=prompt)])

    return {
        "messages": [response],
        "inventory_data": inventory
    }

async def logistics_agent(state: AgentState):
    items = state.get('requested_items', [])
    location = items[0].get('location', "Global Hub") if items else "Global Hub"

    # Check if any message mentions an order ID for shipping status
    all_content = " ".join([m.content for m in state['messages']])
    order_id_match = re.search(r'ORD-\d+', all_content)
    order_id = order_id_match.group() if order_id_match else "ORD-DEFAULT"

    weather = await weather_impact_analysis(location)
    shipping = await get_shipping_status(order_id)

    prompt = LOGISTICS_PROMPT.format(context=f"Location: {location}, Order ID: {order_id}", weather=weather, shipping=shipping)
    response = await support_llm.ainvoke([SystemMessage(content=prompt)])

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
        reasons.append(f"Weather risk too high ({weather_risk})")

    for req in requested_items:
        sku = req['sku']
        qty_requested = req['quantity']
        inv_item = inventory.get(sku, {})
        qty_available = inv_item.get('quantity', 0)

        if qty_available < (0.1 * qty_requested):
            reasons.append(f"Critical stock for {sku}: requested {qty_requested}, only {qty_available} available")

    solution_summary = f"Inventory: {inventory}\nWeather: {weather_risk}\nRequested: {requested_items}"
    prompt = REFLECTION_PROMPT.format(solution=solution_summary)
    response = await brain_llm.ainvoke([SystemMessage(content=prompt)])

    # Update status based on reasons for internal tracking, though LLM decides the alternative
    status = "VALID" if not reasons else "PROPOSAL"

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
        return ["__end__"] # LangGraph parallel returns should be a list of node names or a special string

    selected_agents = []
    if "sales" in decision: selected_agents.append("Sales")
    if "warehouse" in decision: selected_agents.append("Warehouse")
    if "logistics" in decision: selected_agents.append("Logistics")

    if not selected_agents:
        return ["Sales"]

    return selected_agents

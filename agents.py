import os
import json
import re
import asyncio
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
1. If weather risk factor > 0.7, you MUST return STATUS: PROPOSAL or REJECTED.
2. If stock for any item is < 10% of requested quantity, you MUST return STATUS: PROPOSAL or REJECTED.
3. If neither rule is triggered, the status is 'VALID'.

CRITICAL INSTRUCTIONS FOR PROPOSAL/REJECTED:
If the status is PROPOSAL or REJECTED, you must:
- Explain clearly and in detail the reason (e.g., storm at delivery location, critical lack of stock).
- Offer a CREATIVE logistical solution (e.g., reroute, use a different shipping method, switch warehouses, or suggest a partial delivery).
- The "REASON" and "ALTERNATIVE" fields should be detailed and persuasive.

Output your decision in this exact format:
STATUS: [VALID | PROPOSAL | REJECTED]
REASON: [Reason for proposal or validation]
ALTERNATIVE: [Description of the alternative if STATUS is not VALID]
"""

# --- AGENT FUNCTIONS ---

async def sales_agent(state: AgentState):
    last_message = state['messages'][-1].content

    # 1. First Task: Extract SKUs and quantities using strict JSON output
    # We provide the catalog context directly to the LLM for accurate SKU mapping
    catalog_context = db_manager.catalog

    extraction_prompt = f"""Extract SKUs, quantities and location from this message: '{last_message}'.
    Use this catalog for SKU mapping: {json.dumps(catalog_context)}.

    CRITICAL: Return ONLY a valid JSON list of objects.
    Format: [{{"sku": "SKU_CODE", "quantity": 0, "location": "city or null"}}]"""

    extraction_response = await support_llm.ainvoke([
        SystemMessage(content="You are a data extraction assistant. Return ONLY valid JSON in a strict list format."),
        HumanMessage(content=extraction_prompt)
    ])

    try:
        json_match = re.search(r'\[.*\]', extraction_response.content, re.DOTALL)
        requested = json.loads(json_match.group()) if json_match else []
    except Exception:
        requested = []

    # 2. Support tasks: Get demand history and catalog results for the final summary
    demand_history = {}
    for item in requested:
        sku = item.get('sku')
        if sku:
            demand_history[sku] = await db_manager.get_demand_history(sku)

    # We also keep catalog_results in state for other agents if needed
    catalog_results = await db_manager.query_catalog(last_message)

    prompt = SALES_PROMPT.format(context=last_message, catalog=catalog_results, demand=demand_history)
    response = await support_llm.ainvoke([SystemMessage(content=prompt)])

    return {
        "messages": [response],
        "catalog_results": catalog_results,
        "requested_items": requested,
        "demand_history": [demand_history]
    }

async def warehouse_agent(state: AgentState):
    items = state.get('requested_items', [])

    # Concurrent inventory checks using asyncio.gather
    inventory_tasks = []
    skus = []
    for item in items:
        sku = item.get('sku')
        if sku:
            skus.append(sku)
            inventory_tasks.append(get_inventory_levels(sku))

    inventory_results = await asyncio.gather(*inventory_tasks)
    inventory = {sku: res for sku, res in zip(skus, inventory_results)}

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
        sku = req.get('sku')
        qty_requested = req.get('quantity', 0)
        inv_item = inventory.get(sku, {})
        qty_available = inv_item.get('quantity', 0)

        if qty_available < (0.1 * qty_requested):
            reasons.append(f"Critical stock for {sku}: requested {qty_requested}, only {qty_available} available")

    solution_summary = f"Inventory: {inventory}\nWeather Risk: {weather_risk}\nRequested: {requested_items}"
    prompt = REFLECTION_PROMPT.format(solution=solution_summary)
    response = await brain_llm.ainvoke([SystemMessage(content=prompt)])

    # Extract status from LLM response
    llm_content = response.content.upper()
    if "STATUS: VALID" in llm_content:
        status = "VALID"
    elif "STATUS: REJECTED" in llm_content:
        status = "REJECTED"
    else:
        status = "PROPOSAL"

    return {
        "messages": [response],
        "validation_status": status
    }

async def orchestrator_router(state: AgentState):
    last_message = state['messages'][-1].content

    # Preliminary closure check
    terminal_phrases = ["gracias", "de acuerdo", "perfecto", "entendido", "chau", "adios", "ok"]
    if any(phrase in last_message.lower() for phrase in terminal_phrases):
        return ["__end__"]

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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agents import (
    AgentState,
    sales_agent,
    warehouse_agent,
    logistics_agent,
    validation_node,
    orchestrator_router
)
from langchain_core.messages import HumanMessage
import uvicorn

# Define the graph
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("Sales", sales_agent)
builder.add_node("Warehouse", warehouse_agent)
builder.add_node("Logistics", logistics_agent)
builder.add_node("Validation", validation_node)

# --- Orchestrator Routing Logic ---
# The Orchestrator will be the entry point and decide where to go.

async def orchestrator_node(state: AgentState):
    # This node just acts as a pass-through for the decision logic
    # if we wanted to record the orchestration step in the message history.
    # For now, we'll use the orchestrator_router directly as conditional edges.
    return state

builder.add_node("Orchestrator", orchestrator_node)
builder.set_entry_point("Orchestrator")

def route_from_orchestrator(state: AgentState):
    # This is a synchronous wrapper for the router logic if needed,
    # but LangGraph supports async routers.
    # However, for simplicity in edges, let's use the logic from agents.py
    # We'll use a wrapper since add_conditional_edges expects a function returning a string or list.
    pass

# We will use the orchestrator_router to decide which agents to run.
# To handle multiple agents in parallel from the router, we'd need a more complex setup.
# For this demo, we'll follow a slightly simplified but dynamic routing:
# Orchestrator -> [Sales | Warehouse | Logistics | End]
# If Sales is chosen, it then goes to Warehouse/Logistics if needed, then Validation.

builder.add_conditional_edges(
    "Orchestrator",
    orchestrator_router,
    {
        "Sales": "Sales",
        "Warehouse": "Warehouse",
        "Logistics": "Logistics",
        "End": END
    }
)

# Standard flow if Sales is picked first (it usually is for new orders)
builder.add_edge("Sales", "Warehouse")
builder.add_edge("Sales", "Logistics")
builder.add_edge("Warehouse", "Validation")
builder.add_edge("Logistics", "Validation")

# If Warehouse or Logistics were called directly by Orchestrator
builder.add_edge("Validation", END)

# Compile with memory persistence
memory = MemorySaver()
app_graph = builder.compile(checkpointer=memory)

# FastAPI App
app = FastAPI(title="Argos Supply Chain Multi-Agent System")

class Query(BaseModel):
    user_input: str
    thread_id: str = "default-thread"

@app.post("/query")
async def process_query(query: Query):
    config = {"configurable": {"thread_id": query.thread_id}}
    initial_state = {
        "messages": [HumanMessage(content=query.user_input)],
        "requested_items": [],
        "inventory_data": {},
        "shipping_data": {},
        "weather_data": {},
        "catalog_results": [],
        "demand_history": [],
    }

    try:
        final_state = await app_graph.ainvoke(initial_state, config=config)
        last_msg = final_state["messages"][-1].content
        return {
            "response": last_msg,
            "inventory": final_state.get("inventory_data"),
            "weather": final_state.get("weather_data"),
            "validation": final_state.get("validation_status"),
            "requested_items": final_state.get("requested_items")
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "Argos System Online"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

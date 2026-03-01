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

async def orchestrator_node(state: AgentState):
    # Node to trigger the orchestrator router
    return state

builder.add_node("Orchestrator", orchestrator_node)
builder.set_entry_point("Orchestrator")

# Conditional edges from Orchestrator to any combination of agents
builder.add_conditional_edges(
    "Orchestrator",
    orchestrator_router,
    {
        "Sales": "Sales",
        "Warehouse": "Warehouse",
        "Logistics": "Logistics",
        "__end__": END
    }
)

# After each agent, we want to go to Validation
# However, if we run them in parallel, we need to wait for all to finish.
# In LangGraph, if multiple nodes are triggered from a conditional edge,
# they run in parallel and you can converge them to a single node.

builder.add_edge("Sales", "Validation")
builder.add_edge("Warehouse", "Validation")
builder.add_edge("Logistics", "Validation")

# Final step
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

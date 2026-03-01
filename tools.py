import asyncio
import random
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class InventoryQuery(BaseModel):
    sku: str = Field(description="The SKU of the product to check inventory for")

class ShippingQuery(BaseModel):
    order_id: str = Field(description="The order ID to check shipping status for")

class WeatherQuery(BaseModel):
    location: str = Field(description="The location (city or region) to check weather impact for")

async def get_inventory_levels(sku: str) -> Dict[str, Any]:
    """Check the current inventory levels for a given SKU."""
    try:
        await asyncio.sleep(0.5)  # Simulate network latency
        # Mocked data
        inventory_data = {
            "ELEC-001": {"quantity": 150, "unit": "units", "location": "Warehouse A"},
            "ELEC-002": {"quantity": 5, "unit": "units", "location": "Warehouse B"}, # Low stock for testing
            "ELEC-003": {"quantity": 500, "unit": "units", "location": "Warehouse A"},
        }
        return inventory_data.get(sku, {"error": "SKU not found", "quantity": 0})
    except Exception as e:
        return {"error": f"Failed to get inventory levels: {str(e)}", "quantity": 0}

async def get_shipping_status(order_id: str) -> Dict[str, Any]:
    """Retrieve the current shipping status and estimated delivery for an order."""
    try:
        await asyncio.sleep(0.6)
        statuses = ["In Transit", "Processing", "Delivered", "Delayed"]
        return {
            "order_id": order_id,
            "status": random.choice(statuses),
            "estimated_delivery": "2023-10-27",
            "carrier": "LogiSpeed"
        }
    except Exception as e:
        return {"error": f"Failed to get shipping status: {str(e)}", "status": "Unknown"}

async def weather_impact_analysis(location: str) -> Dict[str, Any]:
    """Analyze weather conditions and return a risk factor (0.1 to 1.0) for logistics."""
    try:
        await asyncio.sleep(0.8)
        risk_factor = round(random.uniform(0.1, 1.0), 2)
        condition = "Clear" if risk_factor < 0.4 else "Rainy" if risk_factor < 0.7 else "Stormy"
        return {
            "location": location,
            "risk_factor": risk_factor,
            "condition": condition,
            "impact_description": f"Weather condition is {condition} with a risk factor of {risk_factor}"
        }
    except Exception as e:
        return {"error": f"Failed to perform weather analysis: {str(e)}", "risk_factor": 0.5, "condition": "Unknown"}

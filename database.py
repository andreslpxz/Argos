import asyncio
from typing import List, Dict, Any, Optional

class DatabaseManager:
    """Mock Database Manager with RAG-like capabilities for electronic components and contracts."""
    def __init__(self, connection_string: str = "mock://localhost:5432/supply_chain"):
        self.connection_string = connection_string
        # Mocked catalog of electronic components
        self.catalog = [
            {"sku": "ELEC-001", "name": "Microcontroller STM32", "description": "High-performance ARM Cortex-M4 32-bit RISC core", "price": 4.50, "category": "ICs"},
            {"sku": "ELEC-002", "name": "Resistor 10k Ohm", "description": "Precision 0.1% tolerance thin film resistor", "price": 0.05, "category": "Passives"},
            {"sku": "ELEC-003", "name": "Capacitor 100uF", "description": "Aluminum electrolytic capacitor 50V", "price": 0.12, "category": "Passives"},
            {"sku": "ELEC-004", "name": "Power MOSFET N-Ch", "description": "60V 30A N-Channel power MOSFET in TO-220", "price": 1.20, "category": "Discrete Semis"},
        ]
        # Mocked contracts with delay penalties
        self.contracts = [
            {"id": "CONT-X1", "partner": "SemiconGlobal", "terms": "24h delay penalty: 5% discount per day", "min_order": 100},
            {"id": "CONT-Y2", "partner": "LogiSpeed", "terms": "Guaranteed 48h delivery or 10% refund", "service_level": "Premium"},
        ]

    async def query_catalog(self, search_query: str) -> List[Dict[str, Any]]:
        """Simulate a vector search for catalog items."""
        await asyncio.sleep(0.3)
        # Simple string matching to simulate vector search for this demo
        results = [item for item in self.catalog if search_query.lower() in item['name'].lower() or search_query.lower() in item['description'].lower()]
        return results if results else self.catalog[:2] # Return some defaults if no match

    async def get_contract_details(self, partner_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve contract terms for a specific partner."""
        await asyncio.sleep(0.2)
        for contract in self.contracts:
            if partner_id.lower() in contract['partner'].lower():
                return contract
        return None

    async def get_demand_history(self, sku: str) -> List[Dict[str, Any]]:
        """Retrieve demand history for a given SKU (Sales Agent requirement)."""
        await asyncio.sleep(0.4)
        # Mock demand history
        return [
            {"date": "2023-08", "quantity": 1200},
            {"date": "2023-09", "quantity": 1150},
            {"date": "2023-10", "quantity": 1300},
        ]

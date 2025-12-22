import os
import json
from google import genai
from google.genai import types
import infrastructure.config as config

class GeminiAgent:
    def __init__(self):
        if not config.GENAI_API_KEY:
            raise ValueError("❌ GENAI_API_KEY missing in config.json")
        self.client = genai.Client(api_key=config.GENAI_API_KEY)
        self.model_id = getattr(config, "GENAI_MODEL", "gemini-2.5-pro")
        print(f"   🤖 Agent initialized with model: {self.model_id}")

    def analyze_fundamentals(self, symbol):
        """
        STAGE 1: Universal Financial Health Check.
        No Sector info here. Just Growth, ROE, Debt.
        """
        schema = {
            "type": "OBJECT",
            "properties": {
                "financials": {
                    "type": "OBJECT",
                    "properties": {
                        "sales_growth_3yr_avg": {"type": "NUMBER"},
                        "profit_growth_3yr_avg": {"type": "NUMBER"},
                        "roe_latest": {"type": "NUMBER"},
                        "debt_to_equity": {"type": "NUMBER"},
                        "beta": {"type": "NUMBER"}
                    },
                    "required": ["sales_growth_3yr_avg", "roe_latest", "debt_to_equity"]
                },
                "dcf_inputs": {
                    "type": "OBJECT",
                    "properties": {
                        "free_cash_flow_latest_cr": {"type": "NUMBER"},
                        "growth_rate_projection": {"type": "NUMBER"},
                        "shares_outstanding_cr": {"type": "NUMBER"},
                        "net_debt_cr": {"type": "NUMBER"},
                        "tax_rate": {"type": "NUMBER"}
                    },
                    "required": ["free_cash_flow_latest_cr", "growth_rate_projection"]
                },
                "qualitative": {
                    "type": "OBJECT",
                    "properties": {
                        "management_integrity_score": {"type": "NUMBER"},
                        "reasoning": {"type": "STRING"}
                    }
                }
            },
            "required": ["financials", "dcf_inputs", "qualitative"]
        }

        prompt = f"""
        Analyze Indian stock '{symbol}' (NSE).
        Task: Extract UNIVERSAL fundamental metrics.
        - 3-Year Sales/Profit Growth (CAGR).
        - ROE, Debt/Equity.
        - DCF Inputs (FCF, Shares, Net Debt).
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_id, contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    response_mime_type="application/json", response_schema=schema
                )
            )
            return json.loads(response.text)
        except: return None

    def analyze_sector_specifics(self, symbol):
        """
        STAGE 2: Deep Sector Analysis (Varsity Logic).
        """
        schema = {
            "type": "OBJECT",
            "properties": {
                "broad_sector": {
                    "type": "STRING",
                    "description": "One of: [BANK, IT, AUTO, FMCG, PHARMA, ENERGY, METAL, CEMENT, INFRA, CONSUMER, FINANCE]"
                },
                "niche_industry": {"type": "STRING"},
                "sector_kpis": {
                    "type": "OBJECT",
                    "description": "Specific KPIs like NIM for Banks, SSSG for Retail, Deal Wins for IT",
                    "properties": {
                        "kpi_1": {"type": "STRING", "description": "Name: Value (e.g., 'NIM: 3.5%')"},
                        "kpi_2": {"type": "STRING"},
                        "kpi_3": {"type": "STRING"}
                    }
                },
                "moat_rating": {"type": "STRING", "enum": ["Wide", "Narrow", "None"]},
                "competitive_position": {
                    "type": "STRING",
                    "enum": ["LEADER", "CHALLENGER", "LAGGARD"]
                }
            },
            "required": ["broad_sector", "sector_kpis", "competitive_position"]
        }

        prompt = f"""
        Perform a SECTOR-SPECIFIC analysis for '{symbol}' (NSE).
        
        Step 1: Identify the Broad Sector and Niche Industry.
        
        Step 2: Extract the 3 most critical KPIs based on standard Equity Research:
        - BANKS: Gross NPA%, Net Interest Margin (NIM), CASA Ratio.
        - IT: Attrition Rate, Deal Wins (TCV), Revenue/Employee.
        - AUTO: Volume Growth, Margin per Vehicle.
        - CEMENT/STEEL: Capacity Utilization, EBITDA/Tonne.
        - RETAIL/FMCG: Inventory Turnover, Same Store Sales Growth (SSSG).
        
        Step 3: Classify as LEADER (Top 1-2), CHALLENGER, or LAGGARD.
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_id, contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    response_mime_type="application/json", response_schema=schema
                )
            )
            return json.loads(response.text)
        except: return None

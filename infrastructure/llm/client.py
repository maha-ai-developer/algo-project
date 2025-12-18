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
        # Default to flash for speed, or pro for reasoning
        self.model_id = getattr(config, "GENAI_MODEL", "gemini-2.0-flash")
        print(f"   🤖 Agent initialized with model: {self.model_id}")

    def analyze_company(self, symbol):
        """
        Fetches Fundamentals AND Sector in a single AI call.
        """
        print(f"   🔎 Researching {symbol}...")

        # 1. Update Schema to include SECTOR
        schema = {
            "type": "OBJECT",
            "properties": {
                "company_name": {"type": "STRING"},
                "sector": {
                    "type": "STRING", 
                    "enum": ["BANK", "IT", "AUTO", "FMCG", "PHARMA", "ENERGY", "METAL", "FINANCE", "CONSUMER", "INFRA", "TELECOM", "REALTY", "CHEMICALS", "OTHERS"],
                    "description": "Primary business sector of the company."
                },
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
                        "moat_rating": {"type": "STRING", "enum": ["Wide", "Narrow", "None"]},
                        "reasoning": {"type": "STRING"}
                    }
                }
            },
            "required": ["company_name", "sector", "financials", "dcf_inputs", "qualitative"]
        }

        # 2. Update Prompt
        prompt = f"""
        Perform a fundamental analysis of the Indian stock '{symbol}' (NSE).
        Use Google Search to find the latest Annual Report data, Screener.in data, and recent >
		
		Task 1: Identify the Company Name and its Primary SECTOR
        Task 2: Extract Key Ratios (Sales Growth, ROE, Debt/Equity).
        Task 3: Extract DCF Inputs (Latest Free Cash Flow, Outstanding Shares, Net Debt).
        Task 4: Assess Qualitative factors (Management scandals? Competitive Advantage?).

        Important:
        - Return all numbers in CRORES where applicable.
        - Growth rates and Ratios should be decimals (e.g., 20% = 0.20).
        - Be conservative in growth projections.
		- Be accurate with the Sector
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    response_mime_type="application/json",
                    response_schema=schema
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"❌ AI Error for {symbol}: {e}")
            return None

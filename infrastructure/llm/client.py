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
        
        # Load Model from Config, default to 2.0-flash if missing
        self.model_id = getattr(config, "GENAI_MODEL", "gemini-2.5-pro")
        print(f"   🤖 Agent initialized with model: {self.model_id}")

    def analyze_company(self, symbol):
        """
        Performs Deep Research using Google Search to fetch fundamental data.
        Returns a structured JSON matching your Strategy requirements.
        """
        print(f"   🔎 Researching {symbol} via {self.model_id}...")

        # 1. Define the Output Schema (Strict JSON)
        schema = {
            "type": "OBJECT",
            "properties": {
                "financials": {
                    "type": "OBJECT",
                    "properties": {
                        "sales_growth_3yr_avg": {"type": "NUMBER", "description": "Average sales growth over last 3 years as a decimal (e.g. 0.15)"},
                        "profit_growth_3yr_avg": {"type": "NUMBER"},
                        "roe_latest": {"type": "NUMBER", "description": "Return on Equity as decimal"},
                        "debt_to_equity": {"type": "NUMBER"},
                        "beta": {"type": "NUMBER", "description": "Current stock beta"},
                        "interest_coverage_ratio": {"type": "NUMBER"}
                    },
                    "required": ["sales_growth_3yr_avg", "roe_latest", "debt_to_equity"]
                },
                "dcf_inputs": {
                    "type": "OBJECT",
                    "properties": {
                        "free_cash_flow_latest_cr": {"type": "NUMBER", "description": "Latest FCFF in Crores"},
                        "growth_rate_projection": {"type": "NUMBER", "description": "Conservative growth rate for next 5 years (decimal)"},
                        "shares_outstanding_cr": {"type": "NUMBER", "description": "Number of shares in Crores"},
                        "net_debt_cr": {"type": "NUMBER", "description": "Total Debt minus Cash in Crores"},
                        "tax_rate": {"type": "NUMBER"}
                    },
                    "required": ["free_cash_flow_latest_cr", "growth_rate_projection"]
                },
                "qualitative": {
                    "type": "OBJECT",
                    "properties": {
                        "management_integrity_score": {"type": "NUMBER", "description": "Score 1-10 based on governance history"},
                        "moat_rating": {"type": "STRING", "enum": ["Wide", "Narrow", "None"]},
                        "reasoning": {"type": "STRING", "description": "Brief explanation of the rating"}
                    }
                }
            }
        }

        # 2. The Prompt
        prompt = f"""
        Perform a fundamental analysis of the Indian stock '{symbol}' (NSE).
        Use Google Search to find the latest Annual Report data, Screener.in data, and recent news.
        
        Task 1: Extract Key Ratios (Sales Growth, ROE, Debt/Equity).
        Task 2: Extract DCF Inputs (Latest Free Cash Flow, Outstanding Shares, Net Debt).
        Task 3: Assess Qualitative factors (Management scandals? Competitive Advantage?).
        
        Important:
        - Return all numbers in CRORES where applicable.
        - Growth rates and Ratios should be decimals (e.g., 20% = 0.20).
        - Be conservative in growth projections.
        """

        try:
            # 3. Call Gemini with Search Tool
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    response_mime_type="application/json",
                    response_schema=schema
                )
            )
            
            # 4. Parse JSON
            data = json.loads(response.text)
            return data
            
        except Exception as e:
            print(f"❌ AI Error for {symbol}: {e}")
            return None

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
                        "roce_latest": {"type": "NUMBER"},
                        "pe_ratio": {"type": "NUMBER"},
                        "debt_to_equity": {"type": "NUMBER"},
                        "beta": {"type": "NUMBER"}
                    },
                    "required": ["sales_growth_3yr_avg", "roe_latest", "roce_latest", "pe_ratio", "debt_to_equity"]
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
        - ROE, ROCE, P/E Ratio, Debt/Equity.
        - DCF Inputs (FCF, Shares, Net Debt).
        Ref: Fundamental-Analysis.pdf
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
        except Exception as e:
            # Log error type for debugging (don't print every time - too noisy)
            return None

    def monitor_corporate_actions(self, symbols: list):
        """
        PHASE 11: Corporate Actions Monitor.
        
        Uses Google Search to detect critical corporate actions:
        - Dividends (affects pricing)
        - Stock splits/bonuses
        - Acquisitions/Mergers
        - Rights issues
        - Delistings
        
        Returns alerts for active positions.
        """
        schema = {
            "type": "OBJECT",
            "properties": {
                "alerts": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "symbol": {"type": "STRING"},
                            "action_type": {
                                "type": "STRING",
                                "enum": ["DIVIDEND", "SPLIT", "BONUS", "ACQUISITION", "RIGHTS", "DELISTING", "OTHER"]
                            },
                            "severity": {
                                "type": "STRING",
                                "enum": ["CRITICAL", "WARNING", "INFO"]
                            },
                            "headline": {"type": "STRING"},
                            "date": {"type": "STRING", "description": "Date of action (YYYY-MM-DD or 'upcoming')"},
                            "impact": {"type": "STRING", "description": "Expected impact on trading"},
                            "recommendation": {"type": "STRING", "description": "Suggested action"}
                        },
                        "required": ["symbol", "action_type", "severity", "headline", "recommendation"]
                    }
                },
                "scan_timestamp": {"type": "STRING"},
                "total_alerts": {"type": "INTEGER"}
            },
            "required": ["alerts", "total_alerts"]
        }
        
        symbols_str = ", ".join(symbols)
        
        prompt = f"""
        URGENT: Scan for CRITICAL corporate actions affecting these Indian stocks (NSE):
        {symbols_str}
        
        Search for RECENT news (last 7 days) and UPCOMING events:
        
        1. DIVIDENDS: Ex-dividend dates, dividend amounts
        2. STOCK SPLITS: Announced or upcoming splits  
        3. BONUS ISSUES: Record dates, bonus ratios
        4. ACQUISITIONS/MERGERS: Any M&A activity
        5. RIGHTS ISSUES: Rights offerings
        6. DELISTINGS: Any delisting notices
        7. EARNINGS: Major earnings surprises
        8. REGULATORY: SEBI actions, investigations
        
        For each alert, classify severity:
        - CRITICAL: Exit trade immediately (delisting, major crash, acquisition)
        - WARNING: Consider adjusting position (dividend ex-date, split)
        - INFO: Monitor closely (earnings, minor news)
        
        Only return ACTUAL news found. If no corporate actions found, return empty alerts array.
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
            result = json.loads(response.text)
            result['scan_timestamp'] = __import__('datetime').datetime.now().isoformat()
            return result
        except Exception as e:
            print(f"⚠️ Corporate actions scan failed: {e}")
            return {"alerts": [], "total_alerts": 0, "error": str(e)}

    def scan_position_news(self, symbol: str, position_side: str = "LONG"):
        """
        Scan for critical news affecting a specific position.
        
        Returns actionable intelligence for risk management.
        """
        schema = {
            "type": "OBJECT",
            "properties": {
                "symbol": {"type": "STRING"},
                "sentiment": {"type": "STRING", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
                "risk_level": {"type": "STRING", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "key_news": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "headline": {"type": "STRING"},
                            "source": {"type": "STRING"},
                            "impact": {"type": "STRING", "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL"]}
                        }
                    }
                },
                "recommendation": {
                    "type": "STRING",
                    "description": "HOLD, EXIT, or MONITOR with reasoning"
                },
                "volatility_alert": {"type": "BOOLEAN"}
            },
            "required": ["symbol", "sentiment", "risk_level", "recommendation"]
        }
        
        prompt = f"""
        Analyze CURRENT market news for Indian stock '{symbol}' (NSE).
        
        I have a {position_side} position. Check for:
        
        1. Breaking news (last 24 hours)
        2. Analyst upgrades/downgrades
        3. Volume spikes
        4. Sector-wide news affecting the stock
        5. Any warning signs
        
        Provide actionable recommendation:
        - HOLD: No action needed
        - EXIT: Position at risk, consider closing
        - MONITOR: Increased attention required
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
            return {"symbol": symbol, "error": str(e), "recommendation": "MONITOR"}

    def analyze_system_health(self, context: dict) -> dict:
        """
        AI ADVISOR: Analyzes entire StatArb system state and provides recommendations.
        
        This method provides system-wide oversight by analyzing:
        - Pair candidates and quality scores
        - Backtest results (Sharpe, drawdown, win rate)
        - Fundamental report summary
        - Sector distribution
        - Risk metrics
        
        Args:
            context: Dictionary containing system state data
            
        Returns:
            AI analysis with health status, opportunities, risks, and suggestions
        """
        schema = {
            "type": "OBJECT",
            "properties": {
                "system_health": {
                    "type": "STRING",
                    "enum": ["GREEN", "YELLOW", "RED"],
                    "description": "Overall system health status"
                },
                "health_reasoning": {
                    "type": "STRING",
                    "description": "Brief explanation of health status"
                },
                "top_opportunities": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "pair": {"type": "STRING"},
                            "action": {"type": "STRING"},
                            "reasoning": {"type": "STRING"},
                            "priority": {"type": "STRING", "enum": ["HIGH", "MEDIUM", "LOW"]}
                        },
                        "required": ["pair", "action", "reasoning"]
                    },
                    "description": "Top 3 trading opportunities"
                },
                "risk_alerts": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "type": {"type": "STRING"},
                            "description": {"type": "STRING"},
                            "severity": {"type": "STRING", "enum": ["CRITICAL", "WARNING", "INFO"]}
                        },
                        "required": ["type", "description", "severity"]
                    },
                    "description": "Risk alerts and warnings"
                },
                "suggestions": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "Actionable suggestions for the trader"
                },
                "market_outlook": {
                    "type": "STRING",
                    "description": "Brief market context from recent news"
                }
            },
            "required": ["system_health", "health_reasoning", "top_opportunities", "risk_alerts", "suggestions"]
        }
        
        # Format context for prompt
        pairs_summary = context.get('pairs_summary', 'No pair data available')
        backtest_summary = context.get('backtest_summary', 'No backtest data available')
        fundamental_summary = context.get('fundamental_summary', 'No fundamental data available')
        sector_summary = context.get('sector_summary', 'No sector data available')
        
        prompt = f"""
        You are an AI Trading Advisor for a Statistical Arbitrage (Pair Trading) system on Indian NSE stocks.
        
        SYSTEM STATE:
        
        1. PAIR CANDIDATES:
        {pairs_summary}
        
        2. BACKTEST RESULTS:
        {backtest_summary}
        
        3. FUNDAMENTAL HEALTH:
        {fundamental_summary}
        
        4. SECTOR DISTRIBUTION:
        {sector_summary}
        
        TASK:
        Analyze the entire system and provide:
        1. Overall HEALTH STATUS (GREEN/YELLOW/RED)
        2. TOP 3 TRADING OPPORTUNITIES with reasoning
        3. RISK ALERTS that need attention
        4. ACTIONABLE SUGGESTIONS for improvement
        5. Brief MARKET OUTLOOK from current Indian market news
        
        Be specific and actionable. Reference actual pair names and metrics where possible.
        Focus on pairs with high Z-scores, good Sharpe ratios, and strong fundamentals.
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
            result = json.loads(response.text)
            result['_timestamp'] = __import__('datetime').datetime.now().isoformat()
            return result
        except Exception as e:
            return {
                "system_health": "YELLOW",
                "health_reasoning": f"AI analysis failed: {str(e)}",
                "top_opportunities": [],
                "risk_alerts": [{"type": "SYSTEM", "description": f"AI error: {e}", "severity": "WARNING"}],
                "suggestions": ["Check API connectivity", "Retry analysis"],
                "error": str(e)
            }

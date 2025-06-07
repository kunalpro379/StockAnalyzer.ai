import os
import re
import json
import yfinance as yf
from agno.agent import Agent
from agno.models.google import Gemini
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Set environment variable for Google API
os.environ["GOOGLE_API_KEY"] = "AIzaSyCr35hxFrpVsbNWgqOwU6PwmkpwLmO2dJA"

# ================================ AI AGENTS SETUP ================================ #

# Context Understanding Agent - The brain that understands user intent
context_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Advanced context understanding agent that interprets user queries and determines analysis requirements.",
    instructions=[
        "Analyze user queries to understand their specific investment needs and goals.",
        "Extract relevant information: investment amount, time horizon, risk tolerance, expected returns, market preferences.",
        "Identify if the user mentioned specific stocks, sectors, or just general investment goals.",
        "If insufficient information is provided, identify what additional details are needed.",
        "Understand context about Indian stocks, US stocks, or other markets mentioned.",
        "Detect unrealistic expectations (like 20% daily returns) and provide education.",
        "Always respond in a helpful, educational manner even for unrealistic requests."
    ],
    markdown=True
)

# Research Agent for web search and stock discovery
research_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Expert research agent that finds suitable stocks based on user criteria and market research.",
    instructions=[
        "Based on user investment criteria, suggest appropriate stocks and investment strategies.",
        "For Indian market requests, suggest NSE/BSE listed stocks with .NS/.BO suffixes for Yahoo Finance.",
        "Provide realistic return expectations and investment advice.",
        "Educate users about risk management and realistic investment returns.",
        "Suggest diversified portfolios based on user's risk profile and investment amount.",
        "Provide alternative strategies if user's expectations are unrealistic."
    ],
    markdown=True
)

# Financial Analysis Agent
financial_analyst = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Expert financial analyst specializing in fundamental analysis and financial ratios.",
    instructions=[
        "Analyze financial ratios: P/E, P/B, P/S, Debt/Equity, ROE, ROA, Current Ratio.",
        "Evaluate company financial health and valuation metrics.",
        "Compare financial metrics against industry averages and competitors.",
        "Assess financial strength, liquidity, profitability, and efficiency ratios.",
        "Provide insights on whether a stock is undervalued or overvalued based on fundamentals."
    ],
    markdown=True
)

# Technical Analysis Agent
technical_analyst = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Expert technical analyst specializing in chart patterns, indicators, and price action.",
    instructions=[
        "Analyze price charts and identify technical patterns (head & shoulders, cup & handle, triangles, etc.).",
        "Determine support and resistance levels based on historical price data.",
        "Analyze volume patterns and their relationship with price movements.",
        "Calculate and interpret technical indicators (RSI, MACD, Moving Averages, Bollinger Bands).",
        "Provide entry/exit points based on technical analysis.",
        "Assess trend strength and momentum indicators."
    ],
    markdown=True
)

# Earnings Analysis Agent
earnings_analyst = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Specialist in earnings analysis, growth trends, and quarterly performance evaluation.",
    instructions=[
        "Analyze quarterly and annual earnings reports and trends.",
        "Evaluate earnings growth, revenue growth, and margin trends.",
        "Assess earnings quality and sustainability.",
        "Compare actual earnings vs estimates and analyze guidance.",
        "Identify seasonal patterns and cyclical trends in earnings.",
        "Evaluate forward P/E ratios and earnings projections."
    ],
    markdown=True
)

# Dividend Analysis Agent
dividend_analyst = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Expert in dividend analysis, yield calculations, and income investing strategies.",
    instructions=[
        "Analyze dividend history, yield trends, and payout ratios.",
        "Calculate dividend yield, dividend growth rate, and sustainability metrics.",
        "Evaluate dividend coverage ratios and free cash flow.",
        "Assess dividend aristocrats and kings status.",
        "Compare dividend yields across sectors and against treasury rates.",
        "Analyze ex-dividend dates and dividend frequency patterns."
    ],
    markdown=True
)

# Valuation Analysis Agent - NEW ADVANCED AGENT
valuation_analyst = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Expert company valuation specialist using DCF, EV/EBITDA, PEG ratio, and advanced financial modeling.",
    instructions=[
        "Perform comprehensive company valuation using multiple methodologies:",
        "1. DCF Analysis: Calculate intrinsic value using discounted cash flow models with growth assumptions",
        "2. EV/EBITDA Analysis: Compare enterprise value multiples against industry peers and historical averages",
        "3. PEG Ratio Analysis: Evaluate price/earnings relative to growth rates for growth-adjusted valuation",
        "4. Relative Valuation: Compare P/E, P/B, P/S ratios against sector medians and competitors",
        "5. Sum-of-the-Parts: Break down conglomerate valuations by business segments",
        "Determine if stock is undervalued (BUY), overvalued (SELL), or fairly valued (HOLD)",
        "Provide confidence levels and margin-of-safety calculations",
        "Consider qualitative factors: competitive moats, management quality, industry trends",
        "Ideal for value investors seeking fundamental analysis and intrinsic value calculations"
    ],
    markdown=True
)

# Portfolio Management Agent
portfolio_manager = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Advanced portfolio management specialist focusing on optimization and rebalancing strategies.",
    instructions=[
        "Analyze portfolio composition, diversification, and risk metrics.",
        "Suggest optimal portfolio allocation based on risk tolerance and goals.",
        "Identify rebalancing opportunities and timing.",
        "Calculate portfolio beta, Sharpe ratio, and other risk metrics.",
        "Recommend sector allocation and geographic diversification.",
        "Suggest position sizing and risk management strategies."
    ],
    markdown=True
)

# Signal Generation Agent
signal_generator = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="AI-powered signal generation agent that combines multiple analysis methods for trading decisions.",
    instructions=[
        "Generate BUY/SELL/HOLD signals based on comprehensive analysis.",
        "Combine fundamental, technical, and sentiment analysis for signal generation.",
        "Provide confidence levels and risk assessments for each signal.",
        "Consider market conditions, volatility, and macroeconomic factors.",
        "Suggest optimal entry/exit points and position sizing.",
        "Provide stop-loss and take-profit recommendations."
    ],
    markdown=True
)

# Market Data Fetcher - Enhanced to get more comprehensive data
def get_comprehensive_stock_data(symbols, period="1y"):
    """Fetch comprehensive stock data including technical indicators"""
    data = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            
            # Get historical data
            hist = stock.history(period=period)
            if hist.empty:
                continue
                
            # Get stock info
            info = stock.info
            
            # Calculate technical indicators
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
            
            # RSI calculation
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
            bb_std = hist['Close'].rolling(window=20).std()
            hist['BB_Upper'] = hist['BB_Middle'] + (bb_std * 2)
            hist['BB_Lower'] = hist['BB_Middle'] - (bb_std * 2)
            
            # Volume analysis
            hist['Volume_SMA'] = hist['Volume'].rolling(window=20).mean()
            hist['Volume_Ratio'] = hist['Volume'] / hist['Volume_SMA']
            
            data[symbol] = {
                'historical_data': hist,
                'company_info': info,
                'financial_ratios': extract_financial_ratios(info),
                'earnings_data': get_earnings_data(stock),
                'dividend_data': get_dividend_data(stock)
            }
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            continue
    
    return data

def extract_financial_ratios(info):
    """Extract key financial ratios from stock info"""
    return {
        'pe_ratio': info.get('trailingPE', 'N/A'),
        'forward_pe': info.get('forwardPE', 'N/A'),
        'pb_ratio': info.get('priceToBook', 'N/A'),
        'ps_ratio': info.get('priceToSalesTrailing12Months', 'N/A'),
        'debt_to_equity': info.get('debtToEquity', 'N/A'),
        'roe': info.get('returnOnEquity', 'N/A'),
        'roa': info.get('returnOnAssets', 'N/A'),
        'current_ratio': info.get('currentRatio', 'N/A'),
        'quick_ratio': info.get('quickRatio', 'N/A'),
        'gross_margin': info.get('grossMargins', 'N/A'),
        'operating_margin': info.get('operatingMargins', 'N/A'),
        'profit_margin': info.get('profitMargins', 'N/A')
    }

def get_earnings_data(stock):
    """Get earnings data and analysis"""
    try:
        earnings = stock.quarterly_earnings
        return {
            'quarterly_earnings': earnings.to_dict() if not earnings.empty else {},
            'earnings_dates': stock.calendar,
            'earnings_growth': calculate_earnings_growth(earnings) if not earnings.empty else 'N/A'
        }
    except:
        return {'quarterly_earnings': {}, 'earnings_dates': 'N/A', 'earnings_growth': 'N/A'}

def get_dividend_data(stock):
    """Get comprehensive dividend data"""
    try:
        dividends = stock.dividends
        info = stock.info
        
        if len(dividends) > 0:
            annual_dividend = dividends.resample('YE').sum().iloc[-1] if len(dividends) > 0 else 0
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 1))
            dividend_yield = (annual_dividend / current_price) * 100 if current_price > 0 else 0
            
            return {
                'dividend_history': dividends.tail(20).to_dict(),
                'dividend_yield': dividend_yield,
                'annual_dividend': annual_dividend,
                'payout_ratio': info.get('payoutRatio', 'N/A'),
                'dividend_frequency': len(dividends.resample('YE').sum()),
                'dividend_growth_rate': calculate_dividend_growth(dividends)
            }
        else:
            return {'dividend_history': {}, 'dividend_yield': 0, 'annual_dividend': 0}
    except:
        return {'dividend_history': {}, 'dividend_yield': 0, 'annual_dividend': 0}

def calculate_earnings_growth(earnings):
    """Calculate earnings growth rate"""
    try:
        if len(earnings) >= 2:
            recent_earnings = earnings['Earnings'].iloc[-4:].mean()  # Last 4 quarters
            previous_earnings = earnings['Earnings'].iloc[-8:-4].mean()  # Previous 4 quarters
            growth_rate = ((recent_earnings - previous_earnings) / abs(previous_earnings)) * 100
            return round(growth_rate, 2)
        return 'N/A'
    except:
        return 'N/A'

def calculate_dividend_growth(dividends):
    """Calculate dividend growth rate"""
    try:
        annual_dividends = dividends.resample('YE').sum()
        if len(annual_dividends) >= 2:
            recent_div = annual_dividends.iloc[-1]
            previous_div = annual_dividends.iloc[-2]
            growth_rate = ((recent_div - previous_div) / previous_div) * 100
            return round(growth_rate, 2)
        return 'N/A'
    except:
        return 'N/A'

def identify_technical_patterns(hist_data):
    """Identify technical patterns using AI analysis"""
    try:
        # Get recent price data for pattern analysis
        recent_data = hist_data.tail(50)
        
        # Calculate key levels
        resistance_level = recent_data['High'].rolling(window=20).max().iloc[-1]
        support_level = recent_data['Low'].rolling(window=20).min().iloc[-1]
        current_price = recent_data['Close'].iloc[-1]
        
        # Volume analysis
        avg_volume = recent_data['Volume'].mean()
        recent_volume = recent_data['Volume'].iloc[-1]
        volume_spike = recent_volume > (avg_volume * 1.5)
        
        return {
            'support_level': support_level,
            'resistance_level': resistance_level,
            'current_price': current_price,
            'volume_spike': volume_spike,
            'price_trend': 'Bullish' if current_price > recent_data['Close'].rolling(20).mean().iloc[-1] else 'Bearish'
        }
    except:
        return {}

# ================================ ENHANCED CHATBOT CLASS ================================ #

class AdvancedStockAnalyzerChatbot:
    def __init__(self):
        self.conversation_history = []
        self.user_portfolio = {}
        self.analysis_cache = {}
        self.user_context = self.load_user_context()
        
    def load_user_context(self):
        """Load user context from JSON file"""
        try:
            with open('user_context.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'investment_amount': None,
                'risk_tolerance': None,
                'time_horizon': None,
                'market_preference': None,
                'investment_goals': None,
                'previous_queries': []
            }
    
    def save_user_context(self):
        """Save user context to JSON file"""
        try:
            with open('user_context.json', 'w') as f:
                json.dump(self.user_context, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving context: {e}")
        
    def process_query(self, user_query):
        """Enhanced query processing with AI-driven context understanding"""
        print(f"\n🤖 Analyzing your request: '{user_query}'")
        
        # Add query to context
        self.user_context['previous_queries'].append({
            'query': user_query,
            'timestamp': datetime.now().isoformat()
        })
        
        # Use context agent to understand user intent with previous context
        context_prompt = f"""
        User Query: "{user_query}"
        
        Previous Context:
        - Investment Amount: {self.user_context.get('investment_amount', 'Not specified')}
        - Risk Tolerance: {self.user_context.get('risk_tolerance', 'Not specified')}
        - Time Horizon: {self.user_context.get('time_horizon', 'Not specified')}
        - Market Preference: {self.user_context.get('market_preference', 'Not specified')}
        - Investment Goals: {self.user_context.get('investment_goals', 'Not specified')}
        
        Previous Queries: {self.user_context.get('previous_queries', [])}
        
        Analyze this query and:
        1. Extract investment details (amount, timeline, market preference)
        2. Identify if specific stocks were mentioned or if research is needed
        3. Detect unrealistic expectations and provide education
        4. Determine what additional information is needed
        5. Provide structured response with recommendations
        """
        
        try:
            context_analysis = context_agent.run(context_prompt)
            analysis_content = context_analysis.content if hasattr(context_analysis, 'content') else str(context_analysis)
        except Exception as e:
            print(f"Context analysis error: {e}")
            analysis_content = "Unable to analyze context properly."
        
        # Update user context based on analysis
        self.update_user_context(user_query, analysis_content)
        
        # Check if we need to research stocks or if specific stocks were mentioned
        if self.needs_stock_research(user_query, analysis_content):
            return self.handle_research_request(user_query, analysis_content)
        
        # Try to extract specific stock symbols
        symbols = self.extract_specific_symbols(user_query)
        
        if symbols:
            return self.analyze_specific_stocks(symbols, user_query, analysis_content)
        else:
            return self.provide_contextual_guidance(user_query, analysis_content)
    
    def update_user_context(self, query, analysis):
        """Update user context based on AI analysis"""
        try:
            # Extract investment amount
            import re
            amount_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:dollars?|usd|\$|rupees?|inr|₹|crores?|lakhs?)', query.lower())
            if amount_match:
                self.user_context['investment_amount'] = amount_match.group(1)
            
            # Extract time horizon
            if any(word in query.lower() for word in ['day', 'daily', 'tomorrow']):
                self.user_context['time_horizon'] = 'short_term'
            elif any(word in query.lower() for word in ['month', 'weekly', 'weeks']):
                self.user_context['time_horizon'] = 'medium_term'
            elif any(word in query.lower() for word in ['year', 'long', 'retirement']):
                self.user_context['time_horizon'] = 'long_term'
            
            # Extract market preference
            if any(word in query.lower() for word in ['indian', 'india', 'nse', 'bse']):
                self.user_context['market_preference'] = 'indian'
            elif any(word in query.lower() for word in ['us', 'american', 'nasdaq', 'nyse']):
                self.user_context['market_preference'] = 'us'
            
            # Extract risk tolerance from return expectations
            if any(word in query.lower() for word in ['20%', '20 percent', 'high return', '15%']):
                self.user_context['risk_tolerance'] = 'very_high'
            
            self.save_user_context()
        except Exception as e:
            print(f"Error updating context: {e}")
    
    def needs_stock_research(self, query, analysis):
        """Determine if we need to research stocks rather than analyze specific ones"""
        research_keywords = [
            'where to invest', 'what stocks', 'which stocks', 'recommend', 'suggest',
            'best stocks', 'good investment', 'where should i', 'help me find'
        ]
        return any(keyword in query.lower() for keyword in research_keywords)
    
    def extract_specific_symbols(self, query):
        """Extract only specific stock symbols mentioned by user"""
        # Only look for clear stock symbols, not random words
        stock_patterns = [
            r'\b[A-Z]{2,5}\.NS\b',  # Indian stocks with .NS
            r'\b[A-Z]{2,5}\.BO\b',  # Indian stocks with .BO
            r'\bAAPL\b', r'\bTSLA\b', r'\bGOOGL\b', r'\bMSFT\b', r'\bAMZN\b',
            r'\bMETA\b', r'\bNVDA\b', r'\bNFLX\b', r'\bADPE\b'
        ]
        
        symbols = []
        for pattern in stock_patterns:
            symbols.extend(re.findall(pattern, query.upper()))
        
        # Company name mapping (only for well-known companies)
        company_mapping = {
            'APPLE': 'AAPL', 'TESLA': 'TSLA', 'GOOGLE': 'GOOGL',
            'MICROSOFT': 'MSFT', 'AMAZON': 'AMZN', 'META': 'META',
            'NVIDIA': 'NVDA', 'NETFLIX': 'NFLX', 'FACEBOOK': 'META'
        }
        
        query_upper = query.upper()
        for company, symbol in company_mapping.items():
            if company in query_upper:
                symbols.append(symbol)
        
        return list(set(symbols))
    
    def handle_research_request(self, query, analysis):
        """Handle requests where user needs stock recommendations"""
        print("🔍 Researching suitable investment options...")
        
        # Use research agent to find suitable stocks
        research_prompt = f"""
        User Query: "{query}"
        Context Analysis: {analysis}
        User Context: {self.user_context}
        
        Based on the user's investment criteria, provide:
        1. Realistic return expectations education
        2. Suitable stock recommendations for their market preference
        3. Risk warnings and investment education
        4. Specific stock symbols they can analyze further
        5. Portfolio diversification suggestions
        
        For Indian stocks, use proper Yahoo Finance symbols (.NS for NSE, .BO for BSE)
        """
        
        try:
            research_result = research_agent.run(research_prompt)
            research_content = research_result.content if hasattr(research_result, 'content') else str(research_result)
        except Exception as e:
            print(f"Research error: {e}")
            research_content = "Unable to complete research at this time."
        
        return f"""🔍 **Investment Research & Recommendations**

{research_content}

💡 **Next Steps:**
Once you review these recommendations, you can ask me to analyze specific stocks like:
- "Analyze TCS.NS and INFY.NS technical indicators"
- "Compare RELIANCE.NS vs HDFC.NS financial ratios"
- "Show me dividend analysis for WIPRO.NS"

📝 **Your Context Saved:** I've saved your investment preferences for future queries.
"""
    
    def analyze_specific_stocks(self, symbols, query, analysis):
        """Analyze specific stocks mentioned by user"""
        print(f"📊 Analyzing specific stocks: {', '.join(symbols)}...")
        
        # Get comprehensive data
        stock_data = get_comprehensive_stock_data(symbols)
        
        if not stock_data:
            return f"❌ Could not fetch data for {', '.join(symbols)}. Please check the symbols and try again."
        
        # Determine analysis type
        analysis_type = self.determine_analysis_with_ai(query, analysis)
        
        # Perform analysis
        try:
            result = self.perform_comprehensive_analysis(symbols, stock_data, analysis_type, query, analysis)
            
            # Store in conversation history
            self.conversation_history.append({
                'query': query,
                'symbols': symbols,
                'analysis_type': analysis_type,
                'timestamp': datetime.now()
            })
            
            return result
            
        except Exception as e:
            return f"❌ Analysis error: {str(e)}"
    
    def provide_contextual_guidance(self, query, analysis):
        """Provide guidance when no specific stocks mentioned"""
        return f"""🤔 **I understand your investment interest!**

 **Based on your query:** "{query}"
 
 {analysis}

**To help you better, I can:**
1. 📊 **Research suitable stocks** based on your criteria
2. 📈 **Analyze specific stocks** if you have symbols in mind
3. 💰 **Provide portfolio suggestions** for your investment amount
4. 🎯 **Create investment strategy** based on your goals

**Your saved context:**
- Investment Amount: {self.user_context.get('investment_amount', 'Not specified')}
- Time Horizon: {self.user_context.get('time_horizon', 'Not specified')}
- Market Preference: {self.user_context.get('market_preference', 'Not specified')}
- Risk Tolerance: {self.user_context.get('risk_tolerance', 'Not specified')}

**Try asking:**
- "Find me good Indian stocks for ₹2 lakhs investment"
- "What are the best US tech stocks right now?"
- "Analyze RELIANCE.NS financial ratios"
- "Compare TCS.NS vs INFY.NS"
"""
    
    def determine_analysis_with_ai(self, query, context):
        """Use AI to determine the type of analysis needed"""
        query_lower = query.lower()
        
        # Multi-faceted analysis type detection
        analysis_types = []
        
        if any(word in query_lower for word in ['financial', 'ratio', 'pe', 'pb', 'debt', 'equity', 'roe', 'roa', 'fundamental']):
            analysis_types.append('financial')
        
        if any(word in query_lower for word in ['technical', 'chart', 'pattern', 'support', 'resistance', 'rsi', 'macd', 'volume']):
            analysis_types.append('technical')
        
        if any(word in query_lower for word in ['earnings', 'revenue', 'profit', 'quarterly', 'annual', 'growth']):
            analysis_types.append('earnings')
        
        if any(word in query_lower for word in ['dividend', 'yield', 'payout', 'income', 'distribution']):
            analysis_types.append('dividend')
        
        if any(word in query_lower for word in ['portfolio', 'allocation', 'diversification', 'rebalance', 'risk']):
            analysis_types.append('portfolio')
        
        if any(word in query_lower for word in ['signal', 'recommendation', 'buy', 'sell', 'hold', 'trade']):
            analysis_types.append('signals')
        
        return analysis_types if analysis_types else ['comprehensive']
    
    def perform_comprehensive_analysis(self, symbols, stock_data, analysis_types, user_query, context):
        """Perform comprehensive analysis based on determined types"""
        results = []
        
        for analysis_type in analysis_types:
            if analysis_type == 'financial':
                results.append(self.perform_financial_analysis(symbols, stock_data, user_query))
            elif analysis_type == 'technical':
                results.append(self.perform_technical_analysis(symbols, stock_data, user_query))
            elif analysis_type == 'earnings':
                results.append(self.perform_earnings_analysis(symbols, stock_data, user_query))
            elif analysis_type == 'dividend':
                results.append(self.perform_dividend_analysis(symbols, stock_data, user_query))
            elif analysis_type == 'portfolio':
                results.append(self.perform_portfolio_analysis(symbols, stock_data, user_query))
            elif analysis_type == 'signals':
                results.append(self.perform_signal_analysis(symbols, stock_data, user_query))
            else:  # comprehensive
                results.append(self.perform_full_analysis(symbols, stock_data, user_query))
        
        return "\n\n" + "="*80 + "\n\n".join(results)
    
    def perform_financial_analysis(self, symbols, stock_data, user_query):
        """Comprehensive financial ratio analysis"""
        analysis_data = []
        
        for symbol in symbols:
            data = stock_data[symbol]
            ratios = data['financial_ratios']
            info = data['company_info']
            
            analysis_data.append(f"""
**{symbol} - {info.get('longName', 'N/A')}**
📊 **Financial Ratios:**
• P/E Ratio: {ratios['pe_ratio']}
• Forward P/E: {ratios['forward_pe']}
• P/B Ratio: {ratios['pb_ratio']}
• P/S Ratio: {ratios['ps_ratio']}
• Debt/Equity: {ratios['debt_to_equity']}
• ROE: {ratios['roe']}
• ROA: {ratios['roa']}
• Current Ratio: {ratios['current_ratio']}
• Gross Margin: {ratios['gross_margin']}
• Operating Margin: {ratios['operating_margin']}
• Profit Margin: {ratios['profit_margin']}
""")
        
        # Get AI analysis
        financial_analysis = financial_analyst.run(
            f"Analyze these financial ratios for {', '.join(symbols)}:\n" + 
            "\n".join(analysis_data) + 
            f"\nUser query context: {user_query}\n" +
            "Provide comprehensive financial health assessment, valuation analysis, and investment implications."
        )
        
        return f"💰 **FINANCIAL ANALYSIS**\n\n{financial_analysis.content}"
    
    def perform_technical_analysis(self, symbols, stock_data, user_query):
        """Comprehensive technical analysis"""
        technical_data = []
        
        for symbol in symbols:
            data = stock_data[symbol]
            hist = data['historical_data']
            patterns = identify_technical_patterns(hist)
            
            # Get recent technical indicators
            latest = hist.iloc[-1]
            
            technical_data.append(f"""
**{symbol} Technical Data:**
📈 **Price Action:**
• Current Price: ${latest['Close']:.2f}
• Support Level: ${patterns.get('support_level', 'N/A')}
• Resistance Level: ${patterns.get('resistance_level', 'N/A')}
• Trend: {patterns.get('price_trend', 'N/A')}

📊 **Technical Indicators:**
• RSI (14): {latest['RSI']:.2f}
• SMA 20: ${latest['SMA_20']:.2f}
• SMA 50: ${latest['SMA_50']:.2f}
• SMA 200: ${latest['SMA_200']:.2f}
• Bollinger Upper: ${latest['BB_Upper']:.2f}
• Bollinger Lower: ${latest['BB_Lower']:.2f}

📊 **Volume Analysis:**
• Volume Ratio: {latest['Volume_Ratio']:.2f}
• Volume Spike: {patterns.get('volume_spike', False)}
""")
        
        # Get AI technical analysis
        tech_analysis = technical_analyst.run(
            f"Perform technical analysis for {', '.join(symbols)}:\n" + 
            "\n".join(technical_data) + 
            f"\nUser query: {user_query}\n" +
            "Identify patterns, support/resistance levels, trend analysis, and trading opportunities."
        )
        
        return f"📈 **TECHNICAL ANALYSIS**\n\n{tech_analysis.content}"
    
    def perform_earnings_analysis(self, symbols, stock_data, user_query):
        """Comprehensive earnings analysis"""
        earnings_data = []
        
        for symbol in symbols:
            data = stock_data[symbol]
            earnings = data['earnings_data']
            info = data['company_info']
            
            earnings_data.append(f"""
**{symbol} - {info.get('longName', 'N/A')}**
📊 **Earnings Metrics:**
• Earnings Growth: {earnings['earnings_growth']}%
• Forward P/E: {info.get('forwardPE', 'N/A')}
• Trailing P/E: {info.get('trailingPE', 'N/A')}
• EPS (TTM): ${info.get('trailingEps', 'N/A')}
• Revenue Growth: {info.get('revenueGrowth', 'N/A')}
• Quarterly Revenue Growth: {info.get('quarterlyRevenueGrowth', 'N/A')}
• Quarterly Earnings Growth: {info.get('quarterlyEarningsGrowth', 'N/A')}
""")
        
        # Get AI earnings analysis
        earnings_analysis = earnings_analyst.run(
            f"Analyze earnings data for {', '.join(symbols)}:\n" + 
            "\n".join(earnings_data) + 
            f"\nUser query: {user_query}\n" +
            "Evaluate earnings quality, growth trends, and future prospects."
        )
        
        return f"📊 **EARNINGS ANALYSIS**\n\n{earnings_analysis.content}"
    
    def perform_dividend_analysis(self, symbols, stock_data, user_query):
        """Comprehensive dividend analysis"""
        dividend_data = []
        
        for symbol in symbols:
            data = stock_data[symbol]
            dividends = data['dividend_data']
            info = data['company_info']
            
            dividend_data.append(f"""
**{symbol} - {info.get('longName', 'N/A')}**
💰 **Dividend Metrics:**
• Dividend Yield: {dividends['dividend_yield']:.2f}%
• Annual Dividend: ${dividends['annual_dividend']:.2f}
• Payout Ratio: {dividends['payout_ratio']}
• Dividend Growth Rate: {dividends.get('dividend_growth_rate', 'N/A')}%
• Ex-Dividend Date: {info.get('exDividendDate', 'N/A')}
• Dividend Rate: ${info.get('dividendRate', 'N/A')}
""")
        
        # Get AI dividend analysis
        dividend_analysis = dividend_analyst.run(
            f"Analyze dividend data for {', '.join(symbols)}:\n" + 
            "\n".join(dividend_data) + 
            f"\nUser query: {user_query}\n" +
            "Evaluate dividend sustainability, yield attractiveness, and income potential."
        )
        
        return f"💰 **DIVIDEND ANALYSIS**\n\n{dividend_analysis.content}"
    
    def perform_portfolio_analysis(self, symbols, stock_data, user_query):
        """Portfolio optimization and rebalancing analysis"""
        portfolio_data = []
        
        total_market_cap = 0
        for symbol in symbols:
            data = stock_data[symbol]
            info = data['company_info']
            market_cap = info.get('marketCap', 0)
            total_market_cap += market_cap if market_cap else 0
            
            portfolio_data.append(f"""
**{symbol}:**
• Sector: {info.get('sector', 'N/A')}
• Market Cap: ${market_cap:,} if market_cap else 'N/A'
• Beta: {info.get('beta', 'N/A')}
• 52-Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}
• 52-Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}
""")
        
        # Get AI portfolio analysis
        portfolio_analysis = portfolio_manager.run(
            f"Analyze portfolio composition for stocks: {', '.join(symbols)}\n" + 
            "\n".join(portfolio_data) + 
            f"\nUser query: {user_query}\n" +
            "Provide portfolio optimization suggestions, diversification analysis, and rebalancing recommendations."
        )
        
        return f"📈 **PORTFOLIO ANALYSIS**\n\n{portfolio_analysis.content}"
    
    def perform_signal_analysis(self, symbols, stock_data, user_query):
        """AI-powered buy/sell/hold signal generation"""
        signal_data = []
        
        for symbol in symbols:
            data = stock_data[symbol]
            hist = data['historical_data']
            info = data['company_info']
            ratios = data['financial_ratios']
            
            # Get latest technical data
            latest = hist.iloc[-1]
            
            signal_data.append(f"""
**{symbol} Signal Data:**
📊 **Current Metrics:**
• Price: ${latest['Close']:.2f}
• RSI: {latest['RSI']:.2f}
• P/E Ratio: {ratios['pe_ratio']}
• Volume Ratio: {latest['Volume_Ratio']:.2f}
• 20-day trend: {'Bullish' if latest['Close'] > latest['SMA_20'] else 'Bearish'}
• 50-day trend: {'Bullish' if latest['Close'] > latest['SMA_50'] else 'Bearish'}
""")
        
        # Generate AI-powered signals
        signals = signal_generator.run(
            f"Generate trading signals for {', '.join(symbols)}:\n" + 
            "\n".join(signal_data) + 
            f"\nUser query: {user_query}\n" +
            "Provide BUY/SELL/HOLD recommendations with confidence levels, entry/exit points, and risk management."
        )
        
        return f"🚦 **AI TRADING SIGNALS**\n\n{signals.content}"
    
    def perform_full_analysis(self, symbols, stock_data, user_query):
        """Comprehensive analysis combining all methods"""
        # Perform all analysis types
        financial = self.perform_financial_analysis(symbols, stock_data, user_query)
        technical = self.perform_technical_analysis(symbols, stock_data, user_query)
        earnings = self.perform_earnings_analysis(symbols, stock_data, user_query)
        dividend = self.perform_dividend_analysis(symbols, stock_data, user_query)
        signals = self.perform_signal_analysis(symbols, stock_data, user_query)
        
        return f"""🔍 **COMPREHENSIVE STOCK ANALYSIS**

{financial}

{technical}

{earnings}

{dividend}

{signals}

⚠️ **Disclaimer:** This analysis is for informational purposes only. Always conduct your own research and consult with financial advisors before making investment decisions."""



def run_advanced_chatbot():
    """Main enhanced chatbot interface"""
    chatbot = AdvancedStockAnalyzerChatbot()
    
    print("🤖 **AI Stock Investment Advisor**")
    print("=" * 60)
    print("🚀 **I'm your intelligent investment companion!**")
    print("💡 I understand context, save your preferences, and provide personalized advice")
    print("\n✨ **What I can do:**")
    print("🔍 Research stocks based on your criteria")
    print("📊 Analyze specific stocks (financial, technical, earnings, dividends)")
    print("💰 Provide realistic investment strategies")
    print("📈 Generate AI-powered trading signals")
    print("🎯 Save your investment context for future conversations")
    print("\n💬 **Just tell me about your investment goals naturally!**")
    print("Examples:")
    print("- 'I have ₹2 lakhs to invest in Indian stocks for 1 year'")
    print("- 'Find me good dividend stocks for retirement'")
    print("- 'Analyze Apple vs Tesla technical indicators'")
    print("\nType 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thank you for using AI Stock Investment Advisor! Happy investing!")
                print("💾 Your investment context has been saved for next time.")
                break
            
            if not user_input:
                print("Please enter a query or type 'quit' to exit.")
                continue
            
            response = chatbot.process_query(user_input)
            print(f"\n🤖 AI Analyst:\n{response}\n")
            print("-" * 100)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

# Legacy functions maintained for compatibility
def compare_stocks(symbols):
    data = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="6mo")
            if hist.empty:
                print(f"No data found for {symbol}, skipping it.")
                continue
            data[symbol] = hist['Close'].pct_change().sum()
        except Exception as e:
            print(f"Could not retrieve data for {symbol}. Reason: {str(e)}")
            continue
    return data

# Main execution
if __name__ == "__main__":
    run_advanced_chatbot()
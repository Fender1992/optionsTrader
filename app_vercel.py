#!/usr/bin/env python
"""
Vercel-Optimized Trading App
Manual/On-Demand trading instead of continuous automation
"""

import os
from datetime import datetime
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from src.strategies.max_profit_day_trading import MaxProfitDayTradingStrategy
from src.execution.profit_maximizer import ProfitMaximizer

app = FastAPI(title="Maximum Profit Options Trading - Vercel")

class VercelTradingApp:
    """Vercel-optimized trading app for manual execution"""
    
    def __init__(self):
        self.account_size = float(os.getenv('INITIAL_CAPITAL', '10.0'))
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'C1RG6K7KKOHWS5TG')
        self.strategy = MaxProfitDayTradingStrategy(self.account_size)
        
    def get_status(self):
        """Get current trading configuration"""
        config = self.strategy.get_current_config()
        
        daily_profit_potential = (
            config['account_size'] * 
            config['position_size_pct'] * 
            config['profit_target'] * 
            config['trades_per_day']
        )
        
        return {
            'account_size': config['account_size'],
            'daily_trades': config['trades_per_day'],
            'position_size_pct': config['position_size_pct'] * 100,
            'profit_target_pct': config['profit_target'] * 100,
            'stop_loss_pct': config['stop_loss'] * 100,
            'max_concurrent': config['max_concurrent'],
            'target_symbols': config['target_symbols'],
            'daily_profit_potential': daily_profit_potential,
            'current_time': datetime.now().strftime('%H:%M:%S'),
            'deployment': 'vercel'
        }
    
    async def execute_single_trade_cycle(self):
        """Execute a single trade cycle (Vercel-friendly)"""
        profit_maximizer = ProfitMaximizer(self.account_size, self.api_key)
        
        try:
            # Single cycle execution (not continuous loop)
            await profit_maximizer._execute_trading_cycle()
            performance = profit_maximizer.get_performance_summary()
            return {
                'success': True,
                'trades_executed': performance['total_trades'],
                'current_capital': performance['current_capital'],
                'return_pct': performance['total_return_pct']
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Global app instance
trading_app = VercelTradingApp()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Vercel-optimized trading dashboard"""
    status = trading_app.get_status()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Maximum Profit Options Trading - Vercel</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            .container {{ 
                max-width: 400px; 
                margin: 0 auto; 
                background: rgba(255,255,255,0.1); 
                padding: 20px; 
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }}
            .header {{ 
                text-align: center; 
                margin-bottom: 30px;
            }}
            .metric {{ 
                display: flex; 
                justify-content: space-between; 
                margin: 15px 0;
                padding: 10px;
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
            }}
            .btn {{ 
                width: 100%;
                padding: 15px; 
                margin: 10px 0; 
                border: none; 
                border-radius: 10px; 
                font-size: 16px; 
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s;
            }}
            .btn-primary {{ 
                background: #28a745; 
                color: white;
            }}
            .btn-primary:hover {{
                background: #218838;
                transform: translateY(-2px);
            }}
            .symbols {{
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
                justify-content: center;
            }}
            .symbol {{
                background: rgba(255,255,255,0.2);
                color: white;
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 12px;
            }}
        </style>
        <script>
            async function executeTrade() {{
                const btn = document.getElementById('trade-btn');
                btn.disabled = true;
                btn.innerText = 'Executing Trade...';
                
                try {{
                    const response = await fetch('/execute-trade', {{ method: 'POST' }});
                    const result = await response.json();
                    
                    if (result.success) {{
                        alert(`Trade Executed!\\nTrades: ${{result.trades_executed}}\\nCapital: $${{result.current_capital.toFixed(2)}}\\nReturn: ${{result.return_pct.toFixed(1)}}%`);
                    }} else {{
                        alert(`Trade Failed: ${{result.error}}`);
                    }}
                }} catch (error) {{
                    alert(`Error: ${{error.message}}`);
                }}
                
                btn.disabled = false;
                btn.innerText = 'Execute Trade Cycle';
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>🚀 Maximum Profit Trading</h2>
                <p>Powered by Vercel</p>
                <p>Time: {status['current_time']}</p>
            </div>
            
            <div class="metric">
                <span>Account Size:</span>
                <span>${status['account_size']:.2f}</span>
            </div>
            
            <div class="metric">
                <span>Daily Trades:</span>
                <span>{status['daily_trades']}</span>
            </div>
            
            <div class="metric">
                <span>Position Size:</span>
                <span>{status['position_size_pct']:.1f}%</span>
            </div>
            
            <div class="metric">
                <span>Profit Target:</span>
                <span>{status['profit_target_pct']:.1f}%</span>
            </div>
            
            <div class="metric">
                <span>Daily Potential:</span>
                <span>${status['daily_profit_potential']:.2f}</span>
            </div>
            
            <div style="margin: 20px 0;">
                <div style="text-align: center; margin-bottom: 10px;">Target Symbols:</div>
                <div class="symbols">
                    {''.join([f'<span class="symbol">{symbol}</span>' for symbol in status['target_symbols']])}
                </div>
            </div>
            
            <button id="trade-btn" class="btn btn-primary" onclick="executeTrade()">
                Execute Trade Cycle
            </button>
            
            <div style="text-align: center; margin-top: 20px; font-size: 12px; opacity: 0.8;">
                Manual trading mode - Click to execute trades on-demand<br>
                Perfect for mobile use with Vercel deployment
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.post("/execute-trade")
async def execute_trade():
    """Execute a single trade cycle"""
    return await trading_app.execute_single_trade_cycle()

@app.get("/api/status")
async def get_status():
    """Get current status as JSON"""
    return trading_app.get_status()

# Vercel serverless function handler
def handler(request):
    return app(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
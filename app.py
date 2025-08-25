#!/usr/bin/env python
"""
Maximum Profit Options Trading App
Streamlined for explosive account growth through high-frequency day trading
"""

import asyncio
import os
from datetime import datetime
import logging
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.execution.profit_maximizer import ProfitMaximizer
from src.strategies.max_profit_day_trading import MaxProfitDayTradingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Maximum Profit Options Trading")

# Mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates if directory exists  
if os.path.exists("templates"):
    templates = Jinja2Templates(directory="templates")

class MaxProfitTradingApp:
    """Main trading application for maximum profit generation"""
    
    def __init__(self):
        self.account_size = float(os.getenv('INITIAL_CAPITAL', '10.0'))
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY') or os.getenv('ALPHAVANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required. Check .env file.")
        self.strategy = MaxProfitDayTradingStrategy(self.account_size)
        self.profit_maximizer = None
        self.is_running = False
        
    def get_status(self):
        """Get current trading status"""
        config = self.strategy.get_current_config()
        
        # Calculate profit potential
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
            'weekly_target': daily_profit_potential * 5,
            'monthly_target': daily_profit_potential * 22,
            'is_running': self.is_running,
            'current_time': datetime.now().strftime('%H:%M:%S')
        }
        
    async def start_trading(self):
        """Start the maximum profit trading system"""
        if self.is_running:
            return
            
        self.is_running = True
        self.profit_maximizer = ProfitMaximizer(self.account_size, self.api_key)
        
        logger.info(f"🚀 Starting maximum profit trading with ${self.account_size:.2f}")
        
        try:
            await self.profit_maximizer.start_trading()
        except Exception as e:
            logger.error(f"Trading error: {e}")
            self.is_running = False
            
    def stop_trading(self):
        """Stop the trading system"""
        self.is_running = False
        logger.info("⏹️  Trading stopped")

# Global app instance
trading_app = MaxProfitTradingApp()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Main dashboard"""
    status = trading_app.get_status()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Maximum Profit Options Trading</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background: #f5f5f5;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                padding: 20px; 
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .header {{ 
                text-align: center; 
                color: #1a73e8; 
                margin-bottom: 30px;
                border-bottom: 3px solid #1a73e8;
                padding-bottom: 20px;
            }}
            .status-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 20px; 
                margin-bottom: 30px;
            }}
            .status-card {{ 
                background: #f8f9fa; 
                padding: 20px; 
                border-radius: 8px; 
                border-left: 4px solid #1a73e8;
            }}
            .status-card h3 {{ 
                margin-top: 0; 
                color: #333;
            }}
            .metric {{ 
                display: flex; 
                justify-content: space-between; 
                margin: 10px 0;
                padding: 5px 0;
                border-bottom: 1px solid #eee;
            }}
            .metric:last-child {{
                border-bottom: none;
            }}
            .value {{ 
                font-weight: bold; 
                color: #28a745;
            }}
            .controls {{ 
                text-align: center; 
                margin: 30px 0;
            }}
            .btn {{ 
                padding: 15px 30px; 
                margin: 10px; 
                border: none; 
                border-radius: 5px; 
                font-size: 16px; 
                cursor: pointer;
                transition: all 0.3s;
            }}
            .btn-primary {{ 
                background: #28a745; 
                color: white;
            }}
            .btn-primary:hover {{
                background: #218838;
            }}
            .btn-danger {{ 
                background: #dc3545; 
                color: white;
            }}
            .btn-danger:hover {{
                background: #c82333;
            }}
            .running {{ 
                color: #28a745; 
                font-weight: bold;
            }}
            .stopped {{ 
                color: #dc3545; 
                font-weight: bold;
            }}
            .symbols {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }}
            .symbol {{
                background: #1a73e8;
                color: white;
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 12px;
            }}
            .refresh-btn {{
                position: fixed;
                top: 20px;
                right: 20px;
                background: #1a73e8;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
                cursor: pointer;
            }}
        </style>
        <script>
            function refreshPage() {{
                window.location.reload();
            }}
            
            // Auto refresh every 30 seconds
            setInterval(refreshPage, 30000);
            
            async function startTrading() {{
                document.getElementById('start-btn').disabled = true;
                document.getElementById('start-btn').innerText = 'Starting...';
                
                try {{
                    const response = await fetch('/start', {{ method: 'POST' }});
                    if (response.ok) {{
                        setTimeout(refreshPage, 2000);
                    }}
                }} catch (error) {{
                    console.error('Error starting trading:', error);
                    document.getElementById('start-btn').disabled = false;
                    document.getElementById('start-btn').innerText = 'Start Trading';
                }}
            }}
            
            async function stopTrading() {{
                const response = await fetch('/stop', {{ method: 'POST' }});
                if (response.ok) {{
                    setTimeout(refreshPage, 1000);
                }}
            }}
        </script>
    </head>
    <body>
        <button class="refresh-btn" onclick="refreshPage()">🔄 Refresh</button>
        
        <div class="container">
            <div class="header">
                <h1>🚀 MAXIMUM PROFIT OPTIONS TRADING</h1>
                <p>High-Frequency Day Trading for Explosive Account Growth</p>
                <p>Status: <span class="{'running' if status['is_running'] else 'stopped'}">
                    {'🟢 TRADING ACTIVE' if status['is_running'] else '🔴 STOPPED'}
                </span> | Time: {status['current_time']}</p>
            </div>
            
            <div class="status-grid">
                <div class="status-card">
                    <h3>📊 Account Configuration</h3>
                    <div class="metric">
                        <span>Account Size:</span>
                        <span class="value">${status['account_size']:.2f}</span>
                    </div>
                    <div class="metric">
                        <span>Daily Trades:</span>
                        <span class="value">{status['daily_trades']}</span>
                    </div>
                    <div class="metric">
                        <span>Position Size:</span>
                        <span class="value">{status['position_size_pct']:.1f}%</span>
                    </div>
                    <div class="metric">
                        <span>Profit Target:</span>
                        <span class="value">{status['profit_target_pct']:.1f}%</span>
                    </div>
                    <div class="metric">
                        <span>Stop Loss:</span>
                        <span class="value">{status['stop_loss_pct']:.1f}%</span>
                    </div>
                </div>
                
                <div class="status-card">
                    <h3>💰 Profit Potential</h3>
                    <div class="metric">
                        <span>Daily Target:</span>
                        <span class="value">${status['daily_profit_potential']:.2f}</span>
                    </div>
                    <div class="metric">
                        <span>Weekly Target:</span>
                        <span class="value">${status['weekly_target']:.2f}</span>
                    </div>
                    <div class="metric">
                        <span>Monthly Target:</span>
                        <span class="value">${status['monthly_target']:.2f}</span>
                    </div>
                    <div class="metric">
                        <span>Max Concurrent:</span>
                        <span class="value">{status['max_concurrent']}</span>
                    </div>
                </div>
                
                <div class="status-card">
                    <h3>🎯 Target Instruments</h3>
                    <div class="symbols">
                        {''.join([f'<span class="symbol">{symbol}</span>' for symbol in status['target_symbols']])}
                    </div>
                </div>
                
                <div class="status-card">
                    <h3>⏰ Trading Windows</h3>
                    <div class="metric">
                        <span>09:30-10:30:</span>
                        <span class="value">Opening (4 trades)</span>
                    </div>
                    <div class="metric">
                        <span>12:00-13:00:</span>
                        <span class="value">Lunch (2 trades)</span>
                    </div>
                    <div class="metric">
                        <span>14:30-15:30:</span>
                        <span class="value">Power Hour (4 trades)</span>
                    </div>
                    <div class="metric">
                        <span>15:30-16:00:</span>
                        <span class="value">Close (2 trades)</span>
                    </div>
                </div>
            </div>
            
            <div class="controls">
                {'<button id="stop-btn" class="btn btn-danger" onclick="stopTrading()">⏹️ Stop Trading</button>' if status['is_running'] else '<button id="start-btn" class="btn btn-primary" onclick="startTrading()">🚀 Start Maximum Profit Trading</button>'}
            </div>
            
            <div style="text-align: center; margin-top: 30px; padding: 20px; background: #fff3cd; border-radius: 5px;">
                <h4>⚠️ Risk Warnings</h4>
                <p>This system uses aggressive position sizing for maximum profit potential.<br>
                Only trade with capital you can afford to lose.<br>
                Past performance does not guarantee future results.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.post("/start")
async def start_trading():
    """Start trading endpoint"""
    if not trading_app.is_running:
        # Start trading in background
        asyncio.create_task(trading_app.start_trading())
        return {"status": "started"}
    return {"status": "already_running"}

@app.post("/stop") 
async def stop_trading():
    """Stop trading endpoint"""
    trading_app.stop_trading()
    return {"status": "stopped"}

@app.get("/api/status")
async def get_status():
    """Get current status as JSON"""
    return trading_app.get_status()

if __name__ == "__main__":
    import uvicorn
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    print("\n" + "="*60)
    print("MAXIMUM PROFIT OPTIONS TRADING APP")
    print("="*60)
    print("Starting web interface on http://localhost:8080")
    print("Press Ctrl+C to stop")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8080)
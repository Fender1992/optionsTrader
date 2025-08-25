#!/usr/bin/env python
"""
Production Maximum Profit Trading App
24/7 automated trading with real broker integration
"""

import asyncio
import os
import logging
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
import uvicorn

from src.execution.profit_maximizer import ProfitMaximizer
from src.strategies.max_profit_day_trading import MaxProfitDayTradingStrategy

# Production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Maximum Profit Trading - Production")
security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class ProductionTradingApp:
    """Production trading app with full broker integration"""
    
    def __init__(self):
        # Load configuration from environment
        self.account_size = float(os.getenv('INITIAL_CAPITAL', '10.0'))
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'C1RG6K7KKOHWS5TG')
        
        # Broker configuration
        self.broker_env = os.getenv('TRADIER_ENV', 'paper')
        self.account_id = os.getenv('TRADIER_ACCOUNT_ID')
        self.access_token = os.getenv('TRADIER_ACCESS_TOKEN')
        self.execution_mode = os.getenv('EXECUTION_MODE', 'paper')
        
        # Authentication
        self.username = os.getenv('APP_USERNAME', 'admin')
        self.password_hash = os.getenv('APP_PASSWORD_HASH')
        
        # Trading components
        self.strategy = MaxProfitDayTradingStrategy(self.account_size)
        self.profit_maximizer = None
        self.is_running = False
        self.trades_today = 0
        self.current_pnl = 0.0
        
        logger.info(f"Production app initialized with ${self.account_size} capital")
        
    def verify_credentials(self, credentials: HTTPBasicCredentials):
        """Verify user credentials"""
        if credentials.username != self.username:
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
        if self.password_hash:
            if not pwd_context.verify(credentials.password, self.password_hash):
                raise HTTPException(status_code=401, detail="Invalid credentials")
        else:
            # Default password for development
            if credentials.password != "admin":
                raise HTTPException(status_code=401, detail="Invalid credentials")
                
        return credentials.username
    
    async def start_trading(self):
        """Start 24/7 automated trading"""
        if self.is_running:
            return {"status": "already_running"}
            
        if self.execution_mode == 'live' and not (self.account_id and self.access_token):
            raise ValueError("Live trading requires broker API credentials")
            
        self.is_running = True
        self.profit_maximizer = ProfitMaximizer(self.account_size, self.api_key)
        
        logger.info("🚀 Starting 24/7 automated trading")
        
        # Start trading in background
        asyncio.create_task(self._trading_loop())
        
        return {"status": "started", "mode": self.execution_mode}
    
    async def _trading_loop(self):
        """Main 24/7 trading loop"""
        try:
            await self.profit_maximizer.start_trading()
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            self.is_running = False
    
    def stop_trading(self):
        """Stop automated trading"""
        self.is_running = False
        logger.info("⏹️ Trading stopped")
        return {"status": "stopped"}
    
    def get_status(self):
        """Get comprehensive trading status"""
        config = self.strategy.get_current_config()
        
        return {
            'account_size': config['account_size'],
            'execution_mode': self.execution_mode,
            'broker_env': self.broker_env,
            'is_running': self.is_running,
            'trades_today': self.trades_today,
            'current_pnl': self.current_pnl,
            'daily_trades': config['trades_per_day'],
            'position_size_pct': config['position_size_pct'] * 100,
            'profit_target_pct': config['profit_target'] * 100,
            'target_symbols': config['target_symbols'],
            'current_time': datetime.now().strftime('%H:%M:%S'),
            'server_status': 'production'
        }

# Global app instance
trading_app = ProductionTradingApp()

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, credentials: HTTPBasicCredentials = security.Depends(security)):
    """Secure production trading dashboard"""
    
    # Verify authentication
    trading_app.verify_credentials(credentials)
    status = trading_app.get_status()
    
    # Production dashboard with full controls
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Maximum Profit Trading - Production</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                min-height: 100vh;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
            }}
            .header {{ 
                text-align: center; 
                margin-bottom: 30px;
                padding: 20px;
                background: rgba(255,255,255,0.1);
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }}
            .status-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 20px; 
                margin-bottom: 30px;
            }}
            .status-card {{ 
                background: rgba(255,255,255,0.1); 
                padding: 20px; 
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }}
            .metric {{ 
                display: flex; 
                justify-content: space-between; 
                margin: 15px 0;
                padding: 10px 0;
                border-bottom: 1px solid rgba(255,255,255,0.2);
            }}
            .metric:last-child {{ border-bottom: none; }}
            .value {{ 
                font-weight: bold; 
                color: #4ade80;
            }}
            .controls {{ 
                text-align: center; 
                margin: 30px 0;
            }}
            .btn {{ 
                padding: 15px 30px; 
                margin: 10px; 
                border: none; 
                border-radius: 10px; 
                font-size: 16px; 
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s;
            }}
            .btn-primary {{ 
                background: #10b981; 
                color: white;
            }}
            .btn-danger {{ 
                background: #ef4444; 
                color: white;
            }}
            .running {{ 
                color: #4ade80; 
                font-weight: bold;
                animation: pulse 2s infinite;
            }}
            .stopped {{ 
                color: #ef4444; 
                font-weight: bold;
            }}
            @keyframes pulse {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.7; }}
                100% {{ opacity: 1; }}
            }}
            .warning {{
                background: rgba(239, 68, 68, 0.2);
                border: 1px solid #ef4444;
                border-radius: 10px;
                padding: 15px;
                margin: 20px 0;
                text-align: center;
            }}
        </style>
        <script>
            // Auto refresh every 10 seconds
            setInterval(() => window.location.reload(), 10000);
            
            async function startTrading() {{
                try {{
                    const response = await fetch('/start', {{ method: 'POST' }});
                    const result = await response.json();
                    alert(`Trading ${result.status} in ${result.mode} mode`);
                    window.location.reload();
                }} catch (error) {{
                    alert(`Error: ${error.message}`);
                }}
            }}
            
            async function stopTrading() {{
                if (confirm('Are you sure you want to stop automated trading?')) {{
                    try {{
                        const response = await fetch('/stop', {{ method: 'POST' }});
                        alert('Trading stopped');
                        window.location.reload();
                    }} catch (error) {{
                        alert(`Error: ${error.message}`);
                    }}
                }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Maximum Profit Trading</h1>
                <h2>Production Server - 24/7 Automated</h2>
                <p>Status: <span class="{'running' if status['is_running'] else 'stopped'}">
                    {'🟢 TRADING ACTIVE' if status['is_running'] else '🔴 STOPPED'}
                </span></p>
                <p>Mode: <strong>{status['execution_mode'].upper()}</strong> | Time: {status['current_time']}</p>
            </div>
            
            {'<div class="warning"><strong>⚠️ LIVE TRADING ACTIVE</strong><br>Real money at risk - Monitor carefully</div>' if status['execution_mode'] == 'live' else ''}
            
            <div class="status-grid">
                <div class="status-card">
                    <h3>💰 Account Status</h3>
                    <div class="metric">
                        <span>Account Size:</span>
                        <span class="value">${status['account_size']:.2f}</span>
                    </div>
                    <div class="metric">
                        <span>Today's Trades:</span>
                        <span class="value">{status['trades_today']}</span>
                    </div>
                    <div class="metric">
                        <span>Today's P&L:</span>
                        <span class="value" style="color: {'#4ade80' if status['current_pnl'] >= 0 else '#ef4444'}">${status['current_pnl']:+.2f}</span>
                    </div>
                    <div class="metric">
                        <span>Execution Mode:</span>
                        <span class="value">{status['execution_mode'].upper()}</span>
                    </div>
                </div>
                
                <div class="status-card">
                    <h3>📊 Strategy Config</h3>
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
                </div>
                
                <div class="status-card">
                    <h3>🎯 Target Symbols</h3>
                    <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                        {''.join([f'<span style="background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 15px; font-size: 12px;">{symbol}</span>' for symbol in status['target_symbols']])}
                    </div>
                </div>
            </div>
            
            <div class="controls">
                {'<button class="btn btn-danger" onclick="stopTrading()">⏹️ Stop Trading</button>' if status['is_running'] else '<button class="btn btn-primary" onclick="startTrading()">🚀 Start 24/7 Trading</button>'}
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.post("/start")
async def start_trading(credentials: HTTPBasicCredentials = security.Depends(security)):
    """Start automated trading"""
    trading_app.verify_credentials(credentials)
    return await trading_app.start_trading()

@app.post("/stop")
async def stop_trading(credentials: HTTPBasicCredentials = security.Depends(security)):
    """Stop automated trading"""
    trading_app.verify_credentials(credentials)
    return trading_app.stop_trading()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
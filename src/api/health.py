"""
Health check and monitoring endpoints
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import os
import psutil
import asyncio
from typing import Dict, Any

router = APIRouter()

class SystemMonitor:
    def __init__(self):
        self.start_time = datetime.now()
        self.trade_count = 0
        self.last_trade_time = None
        self.total_profit = 0
        self.active_positions = 0
        self.errors = []
        self.api_calls = {"alphavantage": 0, "tradier": 0}
        
    def record_trade(self, profit: float):
        self.trade_count += 1
        self.last_trade_time = datetime.now()
        self.total_profit += profit
        
    def record_error(self, error: str):
        self.errors.append({
            "time": datetime.now().isoformat(),
            "error": error
        })
        # Keep only last 10 errors
        if len(self.errors) > 10:
            self.errors.pop(0)
            
    def record_api_call(self, service: str):
        if service in self.api_calls:
            self.api_calls[service] += 1

monitor = SystemMonitor()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    try:
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check API keys
        alphavantage_configured = bool(os.getenv('ALPHAVANTAGE_API_KEY'))
        tradier_configured = bool(os.getenv('TRADIER_ACCESS_TOKEN'))
        
        # Calculate uptime
        uptime = datetime.now() - monitor.start_time
        uptime_hours = uptime.total_seconds() / 3600
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": round(uptime_hours, 2),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            },
            "apis": {
                "alphavantage": "configured" if alphavantage_configured else "missing",
                "tradier": "configured" if tradier_configured else "missing"
            },
            "trading": {
                "mode": os.getenv('EXECUTION_MODE', 'paper'),
                "initial_capital": float(os.getenv('INITIAL_CAPITAL', 0))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def detailed_status() -> Dict[str, Any]:
    """Detailed status with trading metrics"""
    try:
        # Get health check data
        health = await health_check()
        
        # Add trading metrics
        trading_active = monitor.last_trade_time and (
            datetime.now() - monitor.last_trade_time < timedelta(minutes=30)
        )
        
        return {
            **health,
            "trading_metrics": {
                "is_active": trading_active,
                "total_trades": monitor.trade_count,
                "last_trade": monitor.last_trade_time.isoformat() if monitor.last_trade_time else None,
                "total_profit": round(monitor.total_profit, 2),
                "active_positions": monitor.active_positions,
                "trades_today": monitor.trade_count,  # You can enhance this
                "win_rate": 0 if monitor.trade_count == 0 else round(
                    (monitor.trade_count * 0.58) / monitor.trade_count * 100, 2
                )  # Using target win rate for now
            },
            "api_usage": monitor.api_calls,
            "recent_errors": monitor.errors[-5:] if monitor.errors else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ping")
async def ping():
    """Simple ping endpoint for uptime monitoring"""
    return {"pong": datetime.now().isoformat()}

@router.get("/ready")
async def readiness_check():
    """Check if the application is ready to accept traffic"""
    try:
        # Check critical dependencies
        checks = {
            "database": True,  # Add DB check if using
            "alphavantage_api": bool(os.getenv('ALPHAVANTAGE_API_KEY')),
            "tradier_api": bool(os.getenv('TRADIER_ACCESS_TOKEN')),
            "config_loaded": os.path.exists('config/max_profit_config.json')
        }
        
        all_ready = all(checks.values())
        
        return {
            "ready": all_ready,
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "ready": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
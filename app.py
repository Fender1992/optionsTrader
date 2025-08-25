from fastapi import FastAPI, Request, HTTPException, Depends, Form, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
import pyotp
import os
import json
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
from src.jobs.scheduler import TradingScheduler
from src.alerts.notifier import EquityMilestoneTracker, Notifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Options Trading App")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBasic()

scheduler = TradingScheduler()

SESSIONS = {}

class AuthManager:
    def __init__(self):
        self.username = os.getenv('APP_USERNAME', 'admin@example.com')
        self.password_hash = os.getenv('APP_PASSWORD_HASH', '')
        self.totp_secret = os.getenv('TOTP_SECRET', '')
        self.ip_allowlist = os.getenv('IP_ALLOWLIST', '').split(',') if os.getenv('IP_ALLOWLIST') else []
    
    def verify_password(self, plain_password: str) -> bool:
        if not self.password_hash:
            return plain_password == "admin"
        return pwd_context.verify(plain_password, self.password_hash)
    
    def verify_totp(self, token: str) -> bool:
        if not self.totp_secret:
            return True
        totp = pyotp.TOTP(self.totp_secret)
        return totp.verify(token, valid_window=1)
    
    def verify_ip(self, client_ip: str) -> bool:
        if not self.ip_allowlist:
            return True
        return client_ip in self.ip_allowlist
    
    def create_session(self, username: str) -> str:
        session_id = secrets.token_urlsafe(32)
        SESSIONS[session_id] = {
            'username': username,
            'created': datetime.now(),
            'last_activity': datetime.now()
        }
        return session_id
    
    def verify_session(self, session_id: str) -> Optional[Dict]:
        if session_id not in SESSIONS:
            return None
        
        session = SESSIONS[session_id]
        
        if datetime.now() - session['last_activity'] > timedelta(hours=1):
            del SESSIONS[session_id]
            return None
        
        session['last_activity'] = datetime.now()
        return session

auth_manager = AuthManager()

def get_current_user(request: Request) -> Dict:
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    session = auth_manager.verify_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Session expired")
    
    return session

@app.on_event("startup")
async def startup_event():
    scheduler.start()
    logger.info("Application started")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.stop()
    logger.info("Application stopped")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    try:
        user = get_current_user(request)
        return RedirectResponse(url="/dashboard")
    except:
        return RedirectResponse(url="/login")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    totp_code: str = Form(None)
):
    client_ip = request.client.host
    
    if not auth_manager.verify_ip(client_ip):
        raise HTTPException(status_code=403, detail="IP not allowed")
    
    if username != auth_manager.username:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not auth_manager.verify_password(password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if auth_manager.totp_secret and not auth_manager.verify_totp(totp_code):
        raise HTTPException(status_code=401, detail="Invalid 2FA code")
    
    session_id = auth_manager.create_session(username)
    
    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=3600
    )
    
    return response

@app.get("/logout")
async def logout(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in SESSIONS:
        del SESSIONS[session_id]
    
    response = RedirectResponse(url="/login")
    response.delete_cookie("session_id")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: Dict = Depends(get_current_user)):
    state = scheduler.trading_state
    
    positions = state.get('positions', [])
    account_info = state.get('account_info', {})
    
    # Get milestone info
    milestone_info = scheduler.milestone_tracker.get_next_target()
    
    pnl_data = []
    if os.path.exists(scheduler.pnl_file):
        pnl_df = pd.read_csv(scheduler.pnl_file)
        if not pnl_df.empty:
            pnl_data = pnl_df.tail(30).to_dict('records')
    
    equity_chart = create_equity_chart(pnl_data)
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "positions": positions,
        "account_info": account_info,
        "capital": state.get('capital', 10000),
        "initial_capital": state.get('initial_capital', 10000),
        "kill_switch": os.getenv('KILL_SWITCH', 'false'),
        "execution_mode": os.getenv('EXECUTION_MODE', 'paper'),
        "equity_chart": equity_chart,
        "last_update": state.get('last_update', 'Never'),
        "milestone_info": milestone_info
    })

@app.post("/api/toggle-kill-switch")
async def toggle_kill_switch(user: Dict = Depends(get_current_user)):
    current = os.getenv('KILL_SWITCH', 'false')
    new_value = 'false' if current == 'true' else 'true'
    os.environ['KILL_SWITCH'] = new_value
    
    scheduler._log_action("kill_switch_toggle", {
        "user": user['username'],
        "new_value": new_value
    })
    
    return {"kill_switch": new_value}

@app.post("/api/set-capital")
async def set_capital(
    request: Request,
    user: Dict = Depends(get_current_user)
):
    data = await request.json()
    capital = float(data.get('capital', 10000))
    
    scheduler.trading_state['capital'] = capital
    scheduler._save_state()
    
    scheduler._log_action("capital_update", {
        "user": user['username'],
        "new_capital": capital
    })
    
    return {"capital": capital}

@app.post("/api/toggle-mode")
async def toggle_mode(user: Dict = Depends(get_current_user)):
    current = os.getenv('EXECUTION_MODE', 'paper')
    new_mode = 'live' if current == 'paper' else 'paper'
    os.environ['EXECUTION_MODE'] = new_mode
    
    scheduler._log_action("mode_toggle", {
        "user": user['username'],
        "new_mode": new_mode
    })
    
    return {"execution_mode": new_mode}

@app.get("/api/positions")
async def get_positions(user: Dict = Depends(get_current_user)):
    positions = scheduler.broker.get_positions()
    return {"positions": positions}

@app.get("/api/logs")
async def get_logs(user: Dict = Depends(get_current_user)):
    logs = []
    if os.path.exists(scheduler.log_file):
        with open(scheduler.log_file, 'r') as f:
            logs = json.load(f)
    
    return {"logs": logs[-100:]}

@app.get("/api/pnl")
async def get_pnl(user: Dict = Depends(get_current_user)):
    if os.path.exists(scheduler.pnl_file):
        pnl_df = pd.read_csv(scheduler.pnl_file)
        return {"pnl": pnl_df.tail(100).to_dict('records')}
    return {"pnl": []}

@app.post("/api/run-job")
async def run_job(
    request: Request,
    user: Dict = Depends(get_current_user)
):
    data = await request.json()
    job_name = data.get('job')
    
    if job_name == 'update_data':
        scheduler.update_data()
    elif job_name == 'build_features':
        scheduler.build_features()
    elif job_name == 'generate_signals':
        scheduler.generate_signals()
    elif job_name == 'execute_trades':
        scheduler.execute_trades()
    elif job_name == 'reconcile':
        scheduler.reconcile_positions()
    else:
        raise HTTPException(status_code=400, detail="Invalid job name")
    
    return {"status": "Job started", "job": job_name}

@app.get("/api/alerts/equity-doubling")
async def get_equity_doubling(user: Dict = Depends(get_current_user)):
    milestone_info = scheduler.milestone_tracker.get_next_target()
    return milestone_info

@app.post("/api/alerts/equity-doubling/reset")
async def reset_equity_baseline(
    request: Request,
    user: Dict = Depends(get_current_user)
):
    data = await request.json()
    totp_code = data.get('totp_code')
    
    # Verify TOTP if configured
    if auth_manager.totp_secret and not auth_manager.verify_totp(totp_code):
        raise HTTPException(status_code=401, detail="Invalid 2FA code")
    
    current_equity = scheduler.trading_state.get('account_info', {}).get('total_equity', 
                                                   scheduler.trading_state.get('capital', 10000))
    
    result = scheduler.milestone_tracker.reset_baseline(current_equity)
    
    scheduler._log_action("baseline_reset", {
        "user": user['username'],
        "new_baseline": current_equity
    })
    
    return result

@app.post("/api/alerts/test")
async def test_notification(
    request: Request,
    user: Dict = Depends(get_current_user)
):
    data = await request.json()
    totp_code = data.get('totp_code')
    
    # Verify TOTP if configured
    if auth_manager.totp_secret and not auth_manager.verify_totp(totp_code):
        raise HTTPException(status_code=401, detail="Invalid 2FA code")
    
    notifier = Notifier(scheduler.config)
    success = notifier.send_test_notification()
    
    return {"success": success, "message": "Test notification sent" if success else "Failed to send"}

@app.get("/api/notifications")
async def get_notifications(user: Dict = Depends(get_current_user)):
    notifications_file = "artifacts/live/pwa_notifications.json"
    
    if os.path.exists(notifications_file):
        with open(notifications_file, 'r') as f:
            notifications = json.load(f)
        
        # Return only unread notifications
        unread = [n for n in notifications if not n.get('read', False)]
        return {"notifications": unread}
    
    return {"notifications": []}

@app.post("/api/notifications/mark-read")
async def mark_notifications_read(
    request: Request,
    user: Dict = Depends(get_current_user)
):
    data = await request.json()
    notification_ids = data.get('ids', [])
    
    notifications_file = "artifacts/live/pwa_notifications.json"
    
    if os.path.exists(notifications_file):
        with open(notifications_file, 'r') as f:
            notifications = json.load(f)
        
        for notif in notifications:
            if notif.get('id') in notification_ids:
                notif['read'] = True
        
        with open(notifications_file, 'w') as f:
            json.dump(notifications, f, indent=2)
    
    return {"success": True}

@app.get("/manifest.json")
async def manifest():
    return {
        "name": "Options Trading App",
        "short_name": "Options",
        "description": "Automated options trading platform",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#1a73e8",
        "icons": [
            {
                "src": "/static/icon-192.png",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "/static/icon-512.png",
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    }

def create_equity_chart(pnl_data: list) -> str:
    if not pnl_data:
        return ""
    
    df = pd.DataFrame(pnl_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df['timestamp']),
        y=df['total_equity'],
        mode='lines',
        name='Total Equity',
        line=dict(color='#1a73e8', width=2)
    ))
    
    fig.update_layout(
        title="Portfolio Equity",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        height=400,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
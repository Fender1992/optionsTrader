import os
import json
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict, Optional
import math
import requests

logger = logging.getLogger(__name__)

class EquityMilestoneTracker:
    def __init__(self, config: Dict):
        self.config = config.get('alerts', {}).get('equity_doubling', {})
        self.enabled = self.config.get('enabled', True)
        self.baseline_mode = self.config.get('baseline_mode', 'initial_capital')
        self.custom_baseline = self.config.get('custom_baseline')
        self.channels = self.config.get('channels', ['pwa_push', 'email'])
        self.email_to = self.config.get('email_to')
        self.sms_to = self.config.get('sms_to')
        self.check_freq = self.config.get('check_freq', 'D')
        self.timezone = self.config.get('timezone', 'America/New_York')
        
        self.state_file = "artifacts/live/milestone_state.json"
        self.state = self._load_state()
        
        self.notifier = Notifier(config)
        
    def _load_state(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load milestone state: {e}")
        
        return {
            'equity_baseline': None,
            'highest_equity_seen': 0,
            'last_milestone_power': 0,
            'milestones_hit': [],
            'last_check': None
        }
    
    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save milestone state: {e}")
    
    def initialize_baseline(self, initial_capital: float):
        if self.state['equity_baseline'] is None:
            if self.baseline_mode == 'initial_capital':
                self.state['equity_baseline'] = initial_capital
                logger.info(f"Initialized equity baseline to initial capital: ${initial_capital:,.2f}")
            elif self.baseline_mode == 'custom' and self.custom_baseline:
                self.state['equity_baseline'] = float(self.custom_baseline)
                logger.info(f"Initialized equity baseline to custom value: ${self.custom_baseline:,.2f}")
            else:
                self.state['equity_baseline'] = initial_capital
                logger.info(f"Defaulting equity baseline to initial capital: ${initial_capital:,.2f}")
            
            self._save_state()
    
    def check_milestone(self, current_equity: float) -> Optional[Dict]:
        if not self.enabled:
            return None
        
        if self.state['equity_baseline'] is None:
            logger.warning("Equity baseline not initialized")
            return None
        
        baseline = self.state['equity_baseline']
        
        # Update highest equity
        if current_equity > self.state['highest_equity_seen']:
            self.state['highest_equity_seen'] = current_equity
            logger.info(f"New highest equity: ${current_equity:,.2f}")
        
        # Calculate current milestone power
        if current_equity >= baseline:
            current_milestone_power = int(math.floor(math.log2(current_equity / baseline)))
        else:
            current_milestone_power = -1
        
        # Check if we've hit a new milestone
        if current_milestone_power > self.state['last_milestone_power']:
            multiplier = 2 ** current_milestone_power
            milestone_value = baseline * multiplier
            
            milestone_id = f"2x^{current_milestone_power}"
            
            # Check for duplicate (idempotency)
            if milestone_id not in self.state['milestones_hit']:
                self.state['milestones_hit'].append(milestone_id)
                self.state['last_milestone_power'] = current_milestone_power
                self.state['last_check'] = datetime.now().isoformat()
                self._save_state()
                
                # Prepare notification
                notification = {
                    'milestone_id': milestone_id,
                    'current_equity': current_equity,
                    'baseline': baseline,
                    'multiplier': multiplier,
                    'milestone_value': milestone_value,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Send notifications
                title = f"Equity Milestone Reached: {multiplier}×"
                body = f"Your account equity has reached ${current_equity:,.2f}, which is {multiplier}× your baseline of ${baseline:,.2f}"
                
                self.notifier.notify(self.channels, title, body)
                
                logger.info(f"Milestone reached: {milestone_id} - Equity: ${current_equity:,.2f} ({multiplier}× baseline)")
                
                return notification
        
        self.state['last_check'] = datetime.now().isoformat()
        self._save_state()
        return None
    
    def get_next_target(self) -> Dict:
        if self.state['equity_baseline'] is None:
            return {'next_target': None, 'next_multiplier': None}
        
        next_power = self.state['last_milestone_power'] + 1
        next_multiplier = 2 ** next_power
        next_target = self.state['equity_baseline'] * next_multiplier
        
        return {
            'baseline': self.state['equity_baseline'],
            'highest_equity': self.state['highest_equity_seen'],
            'last_milestone_power': self.state['last_milestone_power'],
            'last_milestone_multiplier': 2 ** self.state['last_milestone_power'] if self.state['last_milestone_power'] > 0 else 1,
            'next_target': next_target,
            'next_multiplier': next_multiplier,
            'milestones_hit': self.state['milestones_hit']
        }
    
    def reset_baseline(self, new_baseline: float):
        self.state['equity_baseline'] = new_baseline
        self.state['last_milestone_power'] = 0
        self.state['milestones_hit'] = []
        self.state['highest_equity_seen'] = new_baseline
        self._save_state()
        
        logger.info(f"Reset equity baseline to ${new_baseline:,.2f}")
        return self.get_next_target()

class Notifier:
    def __init__(self, config: Dict):
        self.config = config
        
        # Email settings from environment
        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER')
        self.smtp_pass = os.getenv('SMTP_PASS')
        self.smtp_from = os.getenv('SMTP_FROM', self.smtp_user)
        
        # PWA push settings
        self.pwa_subscribers = []
        self._load_pwa_subscribers()
        
    def _load_pwa_subscribers(self):
        subscriber_file = "artifacts/live/pwa_subscribers.json"
        if os.path.exists(subscriber_file):
            try:
                with open(subscriber_file, 'r') as f:
                    self.pwa_subscribers = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load PWA subscribers: {e}")
    
    def notify(self, channels: List[str], title: str, body: str):
        for channel in channels:
            try:
                if channel == 'email':
                    self.send_email(title, body)
                elif channel == 'pwa_push':
                    self.send_pwa_push(title, body)
                elif channel == 'sms':
                    self.send_sms(body)
            except Exception as e:
                logger.error(f"Failed to send notification via {channel}: {e}")
    
    def send_email(self, subject: str, body: str):
        if not self.smtp_user or not self.smtp_pass:
            logger.warning("Email credentials not configured")
            return
        
        email_to = self.config.get('alerts', {}).get('equity_doubling', {}).get('email_to')
        if not email_to:
            logger.warning("No email recipient configured")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_from
            msg['To'] = email_to
            msg['Subject'] = f"[Options Trading] {subject}"
            
            html_body = f"""
            <html>
                <body>
                    <h2>{subject}</h2>
                    <p>{body}</p>
                    <hr>
                    <p><small>Sent from Options Trading Platform at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            
            logger.info(f"Email sent to {email_to}: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    def send_pwa_push(self, title: str, body: str):
        # Store notification for PWA to fetch
        notification = {
            'id': datetime.now().timestamp(),
            'title': title,
            'body': body,
            'timestamp': datetime.now().isoformat(),
            'read': False
        }
        
        notifications_file = "artifacts/live/pwa_notifications.json"
        
        try:
            if os.path.exists(notifications_file):
                with open(notifications_file, 'r') as f:
                    notifications = json.load(f)
            else:
                notifications = []
            
            notifications.append(notification)
            
            # Keep only last 100 notifications
            if len(notifications) > 100:
                notifications = notifications[-100:]
            
            os.makedirs(os.path.dirname(notifications_file), exist_ok=True)
            with open(notifications_file, 'w') as f:
                json.dump(notifications, f, indent=2)
            
            logger.info(f"PWA push notification stored: {title}")
            
        except Exception as e:
            logger.error(f"Failed to store PWA notification: {e}")
    
    def send_sms(self, body: str):
        sms_to = self.config.get('alerts', {}).get('equity_doubling', {}).get('sms_to')
        if not sms_to:
            logger.info("SMS not configured")
            return
        
        # Twilio integration (optional)
        twilio_sid = os.getenv('TWILIO_ACCOUNT_SID')
        twilio_token = os.getenv('TWILIO_AUTH_TOKEN')
        twilio_from = os.getenv('TWILIO_FROM_NUMBER')
        
        if not all([twilio_sid, twilio_token, twilio_from]):
            logger.warning("Twilio credentials not configured")
            return
        
        try:
            # Using Twilio API
            url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Messages.json"
            auth = (twilio_sid, twilio_token)
            data = {
                'From': twilio_from,
                'To': sms_to,
                'Body': f"[Options Trading] {body[:160]}"  # SMS limit
            }
            
            response = requests.post(url, auth=auth, data=data)
            
            if response.status_code == 201:
                logger.info(f"SMS sent to {sms_to}")
            else:
                logger.error(f"Failed to send SMS: {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
    
    def send_test_notification(self) -> bool:
        try:
            test_title = "Test Notification"
            test_body = f"This is a test notification sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            channels = self.config.get('alerts', {}).get('equity_doubling', {}).get('channels', [])
            self.notify(channels, test_title, test_body)
            
            return True
        except Exception as e:
            logger.error(f"Test notification failed: {e}")
            return False
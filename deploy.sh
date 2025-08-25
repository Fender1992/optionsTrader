#!/bin/bash
# Production Deployment Script for Maximum Profit Trading

echo "🚀 Deploying Maximum Profit Trading System..."

# Update system
sudo apt update
sudo apt upgrade -y

# Install dependencies
sudo apt install python3 python3-pip git nginx supervisor -y

# Clone/update application
if [ ! -d "/opt/trading" ]; then
    git clone https://github.com/Fender1992/optionsTrader.git /opt/trading
else
    cd /opt/trading && git pull origin main
fi

cd /opt/trading

# Install Python dependencies
pip3 install -r requirements.txt

# Set up environment variables
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "⚠️  Please edit .env file with your broker credentials"
    echo "   nano /opt/trading/.env"
    exit 1
fi

# Create supervisor config for 24/7 running
cat > /etc/supervisor/conf.d/trading.conf << EOF
[program:trading]
command=python3 /opt/trading/app.py
directory=/opt/trading
user=root
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/trading.log
environment=PATH="/usr/bin"
EOF

# Set up nginx reverse proxy
cat > /etc/nginx/sites-available/trading << EOF
server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable nginx config
ln -sf /etc/nginx/sites-available/trading /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Start services
systemctl enable supervisor
systemctl enable nginx
systemctl restart supervisor
systemctl restart nginx

# Create startup script
cat > /opt/trading/start.sh << EOF
#!/bin/bash
cd /opt/trading
export \$(cat .env | xargs)
python3 app.py
EOF

chmod +x /opt/trading/start.sh

echo "✅ Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file: nano /opt/trading/.env"
echo "2. Add your broker API credentials"
echo "3. Set EXECUTION_MODE=live"
echo "4. Restart: supervisorctl restart trading"
echo "5. Access via: http://your_server_ip"
echo ""
echo "Logs: tail -f /var/log/trading.log"
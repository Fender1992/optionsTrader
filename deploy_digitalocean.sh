#!/bin/bash

# DigitalOcean Deployment Script for Options Trading App
# Run this on your DigitalOcean droplet after SSH access

echo "[!] Starting deployment to DigitalOcean..."

# Update system
echo "[+] Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python 3.11 and dependencies
echo "[+] Installing Python and system dependencies..."
sudo apt-get install -y python3.11 python3.11-venv python3-pip git nginx supervisor

# Install Node.js (for any frontend build if needed)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Create app directory
echo "[+] Setting up application directory..."
sudo mkdir -p /opt/optionstrading
sudo chown $USER:$USER /opt/optionstrading
cd /opt/optionstrading

# Clone or pull latest code
if [ -d ".git" ]; then
    echo "[+] Pulling latest code..."
    git pull origin main
else
    echo "[+] Cloning repository..."
    git clone https://github.com/Fender1992/optionsTrader.git .
fi

# Create Python virtual environment
echo "[+] Setting up Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "[+] Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment file (you'll need to upload this separately)
echo "[!] IMPORTANT: Upload your .env file to /opt/optionstrading/.env"
echo "[!] Use: scp .env root@your-server-ip:/opt/optionstrading/.env"

# Create systemd service
echo "[+] Creating systemd service..."
sudo tee /etc/systemd/system/optionstrading.service > /dev/null <<EOF
[Unit]
Description=Options Trading Application
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/optionstrading
Environment="PATH=/opt/optionstrading/venv/bin"
ExecStart=/opt/optionstrading/venv/bin/python /opt/optionstrading/app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Configure Nginx
echo "[+] Configuring Nginx..."
sudo tee /etc/nginx/sites-available/optionstrading > /dev/null <<'EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    location /health {
        proxy_pass http://127.0.0.1:8080/health;
    }
}
EOF

# Enable Nginx site
sudo ln -sf /etc/nginx/sites-available/optionstrading /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

# Enable and start the service
echo "[+] Starting application service..."
sudo systemctl daemon-reload
sudo systemctl enable optionstrading
sudo systemctl start optionstrading

# Set up firewall
echo "[+] Configuring firewall..."
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw --force enable

# Create monitoring script
echo "[+] Creating monitoring script..."
tee /opt/optionstrading/check_status.sh > /dev/null <<'EOF'
#!/bin/bash
echo "=== Options Trading App Status ==="
echo ""
echo "[SERVICE STATUS]"
systemctl status optionstrading --no-pager | head -10
echo ""
echo "[RECENT LOGS]"
journalctl -u optionstrading -n 20 --no-pager
echo ""
echo "[APP HEALTH]"
curl -s http://localhost:8080/health | python3 -m json.tool
EOF
chmod +x /opt/optionstrading/check_status.sh

echo ""
echo "[+] Deployment complete!"
echo ""
echo "=== NEXT STEPS ==="
echo "1. Upload your .env file:"
echo "   scp .env root@your-server-ip:/opt/optionstrading/.env"
echo ""
echo "2. Restart the service:"
echo "   sudo systemctl restart optionstrading"
echo ""
echo "3. Check status:"
echo "   sudo systemctl status optionstrading"
echo "   OR"
echo "   /opt/optionstrading/check_status.sh"
echo ""
echo "4. View logs:"
echo "   sudo journalctl -u optionstrading -f"
echo ""
echo "5. Access your app:"
echo "   http://your-server-ip"
echo ""
echo "6. Check health endpoint:"
echo "   http://your-server-ip/health"
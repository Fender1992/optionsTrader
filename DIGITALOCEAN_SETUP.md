# DigitalOcean Deployment Guide

## Step 1: Create a Droplet

1. Log into DigitalOcean
2. Create a new Droplet:
   - Choose Ubuntu 22.04 LTS
   - Select Basic plan ($6/month minimum)
   - Choose datacenter (preferably close to you)
   - Add SSH key or use password
   - Name: optionstrading

## Step 2: Connect to Your Droplet

```bash
# From your local machine (Windows PowerShell or Terminal)
ssh root@YOUR_DROPLET_IP
```

## Step 3: Quick Deployment

```bash
# Run these commands on your droplet
cd ~
wget https://raw.githubusercontent.com/Fender1992/optionsTrader/main/deploy_digitalocean.sh
chmod +x deploy_digitalocean.sh
./deploy_digitalocean.sh
```

## Step 4: Upload Your Configuration

From your LOCAL machine (Windows):

```powershell
# In PowerShell, from D:\OptionsTrading directory
scp .env root@YOUR_DROPLET_IP:/opt/optionstrading/.env
```

## Step 5: Start the Application

Back on your droplet:

```bash
# Restart the service with your config
sudo systemctl restart optionstrading

# Check if it's running
sudo systemctl status optionstrading
```

## Step 6: Verify Deployment

1. Check service status:
   ```bash
   sudo systemctl status optionstrading
   ```

2. Check application health:
   ```bash
   curl http://localhost:8080/health
   ```

3. View real-time logs:
   ```bash
   sudo journalctl -u optionstrading -f
   ```

4. From your browser:
   - App: http://YOUR_DROPLET_IP
   - Health: http://YOUR_DROPLET_IP/health

## Monitoring Commands

```bash
# Check if app is running
sudo systemctl status optionstrading

# View last 50 log lines
sudo journalctl -u optionstrading -n 50

# Follow logs in real-time
sudo journalctl -u optionstrading -f

# Restart app
sudo systemctl restart optionstrading

# Stop app
sudo systemctl stop optionstrading

# Start app
sudo systemctl start optionstrading

# Check system resources
htop

# Check disk usage
df -h

# Check app health
curl http://localhost:8080/health | python3 -m json.tool
```

## Troubleshooting

### App won't start
```bash
# Check logs for errors
sudo journalctl -u optionstrading -n 100

# Check if port is in use
sudo lsof -i:8080

# Check Python errors
cd /opt/optionstrading
source venv/bin/activate
python app.py
```

### Can't access from browser
```bash
# Check firewall
sudo ufw status

# Check nginx
sudo nginx -t
sudo systemctl restart nginx

# Check if app is listening
curl http://localhost:8080/health
```

### API errors
```bash
# Verify .env file exists and has keys
cat /opt/optionstrading/.env | grep API_KEY
cat /opt/optionstrading/.env | grep TRADIER
```

## Security Setup (Optional but Recommended)

1. Create non-root user:
```bash
adduser tradingapp
usermod -aG sudo tradingapp
```

2. Set up SSL with Let's Encrypt:
```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

3. Enable automatic security updates:
```bash
sudo apt-get install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

## Backup Your Configuration

```bash
# On droplet - backup .env
cp /opt/optionstrading/.env /opt/optionstrading/.env.backup

# From local machine - download backup
scp root@YOUR_DROPLET_IP:/opt/optionstrading/.env ./env_backup_$(date +%Y%m%d)
```

## Auto-Restart on Reboot

The systemd service is already configured to auto-start. To verify:

```bash
sudo systemctl enable optionstrading
sudo systemctl is-enabled optionstrading  # Should show "enabled"
```

## Success Indicators

Your app is successfully deployed when:
- [x] `systemctl status optionstrading` shows "active (running)"
- [x] `http://YOUR_DROPLET_IP/health` returns status "healthy"
- [x] You can login at `http://YOUR_DROPLET_IP`
- [x] Logs show "Application started successfully"

## Production Checklist

- [ ] Change EXECUTION_MODE from "paper" to "live" in .env (when ready)
- [ ] Set up domain name (optional)
- [ ] Configure SSL certificate
- [ ] Set up monitoring alerts
- [ ] Configure backup strategy
- [ ] Test kill switch functionality
# Vigil AI - Deployment Guide

## Overview

This guide covers deploying Vigil AI to a Hostinger VPS with Nginx as reverse proxy.

## Architecture

```
Internet → Nginx (SSL/Reverse Proxy) → Gunicorn (WSGI) → Flask App
                                                            ↓
                                                      Supabase (Master DB)
                                                            ↓
                                                      Client Supabase DBs
```

## Files Created

```
deployment/
├── nginx.conf              # Nginx reverse proxy configuration
├── gunicorn.conf.py        # Gunicorn WSGI server configuration
├── requirements.txt        # Python dependencies
├── DEPLOYMENT.md           # This file
└── hostinger/
    ├── wsgi.py             # WSGI entry point
    ├── vigil.service       # Systemd service file
    ├── setup.sh            # Automated setup script
    └── .htaccess           # Apache config (shared hosting only)
```

## Quick Start (Hostinger VPS)

### 1. SSH into your VPS

```bash
ssh root@your-server-ip
```

### 2. Download setup script

```bash
mkdir -p /var/www/vigil/backend/deployment/hostinger
cd /var/www/vigil/backend/deployment/hostinger
# Upload setup.sh via SFTP or paste contents
chmod +x setup.sh
```

### 3. Run setup script

```bash
./setup.sh
```

### 4. Upload application files

From your local machine:

```bash
# Upload backend code
scp -r Model_Code/Backend/* root@your-server:/var/www/vigil/backend/

# Upload environment file
scp Model_Code/Keys_Security/env root@your-server:/var/www/vigil/Keys_Security/env
```

### 5. Set permissions

```bash
sudo chown -R vigil:www-data /var/www/vigil
sudo chmod 600 /var/www/vigil/Keys_Security/env
```

### 6. Install Python dependencies

```bash
sudo -u vigil /var/www/vigil/venv/bin/pip install -r /var/www/vigil/backend/deployment/requirements.txt
```

### 7. Start services

```bash
sudo systemctl start vigil
sudo systemctl restart nginx
```

### 8. Setup SSL (after DNS configured)

```bash
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

## Manual Setup

### System Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv nginx certbot python3-certbot-nginx git
```

### Create Application User

```bash
sudo useradd -m -s /bin/bash vigil
sudo usermod -aG www-data vigil
```

### Directory Structure

```bash
sudo mkdir -p /var/www/vigil/{backend,frontend,logs,Keys_Security}
sudo mkdir -p /var/log/gunicorn /var/run/gunicorn
sudo chown -R vigil:www-data /var/www/vigil /var/log/gunicorn /var/run/gunicorn
```

### Python Environment

```bash
sudo -u vigil python3 -m venv /var/www/vigil/venv
sudo -u vigil /var/www/vigil/venv/bin/pip install --upgrade pip wheel
sudo -u vigil /var/www/vigil/venv/bin/pip install -r /var/www/vigil/backend/deployment/requirements.txt
```

### Configure Nginx

```bash
sudo cp /var/www/vigil/backend/deployment/nginx.conf /etc/nginx/sites-available/vigil
sudo nano /etc/nginx/sites-available/vigil  # Update domain
sudo ln -sf /etc/nginx/sites-available/vigil /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
```

### Configure Systemd

```bash
sudo cp /var/www/vigil/backend/deployment/hostinger/vigil.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable vigil
```

## Environment Variables

Your `/var/www/vigil/Keys_Security/env` file should contain:

```env
# Master Supabase
MASTER_SUPABASE_URL=https://xxxxx.supabase.co
MASTER_SUPABASE_ANON_KEY=eyJhbGc...
MASTER_SUPABASE_SERVICE_KEY=eyJhbGc...

# Encryption
ENCRYPTION_KEY=your-fernet-key

# X.AI
XAI_API_KEY=your-xai-key

# Flask
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-secret-key

# Email notifications
NOTIFICATION_EMAIL=DB.CoreSight@gmail.com
```

## Service Management

```bash
# Start
sudo systemctl start vigil

# Stop
sudo systemctl stop vigil

# Restart
sudo systemctl restart vigil

# Status
sudo systemctl status vigil

# View logs
sudo journalctl -u vigil -f
```

## Nginx Management

```bash
# Test config
sudo nginx -t

# Reload
sudo systemctl reload nginx

# Restart
sudo systemctl restart nginx

# View logs
sudo tail -f /var/log/nginx/vigil_error.log
sudo tail -f /var/log/nginx/vigil_access.log
```

## SSL Certificate

```bash
# Initial setup
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Renewal (automatic via cron, but manual if needed)
sudo certbot renew

# Check certificate status
sudo certbot certificates
```

## Firewall Setup

```bash
# Allow SSH, HTTP, HTTPS
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw enable
sudo ufw status
```

## Health Checks

```bash
# Check if app is running
curl http://localhost:8000/health

# Check via Nginx
curl https://your-domain.com/health

# Check API
curl https://your-domain.com/api/sync/status
```

## Troubleshooting

### App not starting

```bash
# Check logs
sudo journalctl -u vigil -n 100 --no-pager

# Check Gunicorn directly
cd /var/www/vigil/backend
sudo -u vigil /var/www/vigil/venv/bin/gunicorn --bind 127.0.0.1:8000 wsgi:app
```

### Nginx 502 Bad Gateway

```bash
# Check if Gunicorn is running
sudo systemctl status vigil

# Check socket exists
ls -la /var/run/gunicorn/

# Check Nginx error log
sudo tail -f /var/log/nginx/vigil_error.log
```

### Permission errors

```bash
# Fix ownership
sudo chown -R vigil:www-data /var/www/vigil
sudo chown -R vigil:www-data /var/log/gunicorn
sudo chown -R vigil:www-data /var/run/gunicorn

# Fix env file permissions
sudo chmod 600 /var/www/vigil/Keys_Security/env
```

### Database connection issues

```bash
# Test from server
curl -X GET "https://your-supabase-url.supabase.co/rest/v1/" \
  -H "apikey: your-anon-key"
```

## Updates & Deployments

### Deploy new code

```bash
# Upload new files
scp -r Model_Code/Backend/* root@your-server:/var/www/vigil/backend/

# Restart service
sudo systemctl restart vigil
```

### Zero-downtime deployment

```bash
# Reload workers without downtime
sudo systemctl reload vigil
```

## Monitoring

### Resource usage

```bash
# CPU/Memory
htop

# Disk space
df -h

# Network connections
ss -tlnp
```

### Application logs

```bash
# Real-time logs
sudo journalctl -u vigil -f

# Last 100 lines
sudo journalctl -u vigil -n 100

# Since specific time
sudo journalctl -u vigil --since "1 hour ago"
```

## Backup

### Database (handled by Supabase)

Supabase provides automatic backups. For additional safety:

```bash
# Export via Supabase CLI
supabase db dump -f backup.sql
```

### Application files

```bash
# Backup entire app
tar -czvf vigil-backup-$(date +%Y%m%d).tar.gz /var/www/vigil/
```

## Security Checklist

- [ ] SSL certificate installed and auto-renewal configured
- [ ] Firewall enabled (UFW)
- [ ] Environment file permissions (600)
- [ ] Dedicated application user (non-root)
- [ ] Regular system updates
- [ ] Fail2ban installed (optional but recommended)
- [ ] Log rotation configured

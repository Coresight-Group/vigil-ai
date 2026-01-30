#!/bin/bash
# ============================================================================
# HOSTINGER VPS SETUP SCRIPT FOR VIGIL AI
# Run this script on a fresh Ubuntu/Debian VPS
# ============================================================================

set -e  # Exit on error

echo "============================================"
echo "VIGIL AI - Hostinger VPS Setup"
echo "============================================"

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_DIR="/var/www/vigil"
APP_USER="vigil"

# IMPORTANT: Configure these before running the script!
# Can be set via environment variables or edited directly here
DOMAIN="${VIGIL_DOMAIN:-vigilsecure.com}"
EMAIL="${VIGIL_EMAIL:-admin@vigilsecure.com}"

# Validate domain is configured
if [ "$DOMAIN" = "vigilsecure.com" ] && [ -z "$VIGIL_DOMAIN" ]; then
    echo ""
    echo "WARNING: Using default domain 'vigilsecure.com'"
    echo "Set VIGIL_DOMAIN environment variable or edit this script before production deployment"
    echo ""
    read -p "Continue with default domain? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Set VIGIL_DOMAIN environment variable and re-run:"
        echo "  export VIGIL_DOMAIN=your-actual-domain.com"
        echo "  export VIGIL_EMAIL=your-email@domain.com"
        echo "  ./setup.sh"
        exit 1
    fi
fi

echo "Configuration:"
echo "  Domain: $DOMAIN"
echo "  Email: $EMAIL"
echo "  App Dir: $APP_DIR"
echo ""

# ============================================================================
# SYSTEM UPDATE
# ============================================================================

echo "[1/10] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

echo "[2/10] Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    nginx \
    certbot \
    python3-certbot-nginx \
    git \
    build-essential \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    supervisor \
    curl \
    wget \
    unzip

# ============================================================================
# CREATE APPLICATION USER
# ============================================================================

echo "[3/10] Creating application user..."
if ! id "$APP_USER" &>/dev/null; then
    sudo useradd -m -s /bin/bash $APP_USER
    sudo usermod -aG www-data $APP_USER
fi

# ============================================================================
# CREATE DIRECTORY STRUCTURE
# ============================================================================

echo "[4/10] Creating directory structure..."
sudo mkdir -p $APP_DIR/{backend,frontend,logs,Keys_Security}
sudo mkdir -p /var/log/gunicorn
sudo mkdir -p /var/run/gunicorn

# Set permissions
sudo chown -R $APP_USER:www-data $APP_DIR
sudo chown -R $APP_USER:www-data /var/log/gunicorn
sudo chown -R $APP_USER:www-data /var/run/gunicorn
sudo chmod -R 755 $APP_DIR

# ============================================================================
# SETUP PYTHON VIRTUAL ENVIRONMENT
# ============================================================================

echo "[5/10] Setting up Python virtual environment..."
sudo -u $APP_USER python3 -m venv $APP_DIR/venv

# Activate and install dependencies
sudo -u $APP_USER $APP_DIR/venv/bin/pip install --upgrade pip
sudo -u $APP_USER $APP_DIR/venv/bin/pip install wheel

# ============================================================================
# COPY APPLICATION FILES
# ============================================================================

echo "[6/10] Copying application files..."
echo "Please copy your application files to $APP_DIR/backend/"
echo "You can use SCP, SFTP, or git clone"
echo ""
echo "Example with SCP from local machine:"
echo "  scp -r ./Model_Code/Backend/* user@your-server:$APP_DIR/backend/"
echo "  scp -r ./Model_Code/Keys_Security/* user@your-server:$APP_DIR/Keys_Security/"
echo ""

# ============================================================================
# INSTALL PYTHON DEPENDENCIES
# ============================================================================

echo "[7/10] Installing Python dependencies..."
if [ -f "$APP_DIR/backend/requirements.txt" ]; then
    sudo -u $APP_USER $APP_DIR/venv/bin/pip install -r $APP_DIR/backend/requirements.txt
else
    echo "requirements.txt not found. Installing common dependencies..."
    sudo -u $APP_USER $APP_DIR/venv/bin/pip install \
        flask \
        gunicorn \
        python-dotenv \
        supabase \
        httpx \
        requests \
        cryptography \
        paramiko \
        dropbox \
        google-auth \
        google-auth-oauthlib \
        google-api-python-client \
        psycopg2-binary \
        mysql-connector-python \
        pyodbc \
        PyPDF2 \
        python-docx \
        openpyxl \
        pandas \
        transformers \
        torch \
        numpy \
        apscheduler
fi

# ============================================================================
# SETUP SYSTEMD SERVICE
# ============================================================================

echo "[8/10] Setting up systemd service..."

# Check if vigil.service exists
if [ -f "$APP_DIR/backend/deployment/hostinger/vigil.service" ]; then
    sudo cp $APP_DIR/backend/deployment/hostinger/vigil.service /etc/systemd/system/vigil.service
else
    # Create systemd service file if it doesn't exist
    echo "Creating vigil.service file..."
    sudo tee /etc/systemd/system/vigil.service > /dev/null <<EOF
[Unit]
Description=Vigil AI Risk Management Backend
After=network.target

[Service]
Type=notify
User=$APP_USER
Group=www-data
WorkingDirectory=$APP_DIR/backend
Environment="PATH=$APP_DIR/venv/bin"
EnvironmentFile=$APP_DIR/Keys_Security/env
ExecStart=$APP_DIR/venv/bin/gunicorn -c $APP_DIR/backend/deployment/gunicorn.conf.py app:app
ExecReload=/bin/kill -s HUP \$MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
fi

# Update the service file with correct paths
sudo sed -i "s|/var/www/vigil|$APP_DIR|g" /etc/systemd/system/vigil.service

sudo systemctl daemon-reload
sudo systemctl enable vigil

# ============================================================================
# SETUP NGINX
# ============================================================================

echo "[9/10] Setting up Nginx..."

# Copy nginx config
sudo cp $APP_DIR/backend/deployment/nginx.conf /etc/nginx/sites-available/vigil

# Update domain in config (nginx.conf now uses vigilsecure.com as placeholder)
sudo sed -i "s|vigilsecure.com|$DOMAIN|g" /etc/nginx/sites-available/vigil

# Enable the site
sudo ln -sf /etc/nginx/sites-available/vigil /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx config
sudo nginx -t

# ============================================================================
# SETUP SSL WITH LET'S ENCRYPT
# ============================================================================

echo "[10/10] Setting up SSL certificate..."
echo ""
echo "To obtain SSL certificate, run:"
echo "  sudo certbot --nginx -d $DOMAIN -d www.$DOMAIN --email $EMAIL --agree-tos --non-interactive"
echo ""
echo "Or if DNS is not yet pointing to this server, skip SSL for now."

# ============================================================================
# START SERVICES
# ============================================================================

echo ""
echo "============================================"
echo "SETUP COMPLETE!"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Copy your application files:"
echo "   scp -r ./Model_Code/Backend/* root@your-server:$APP_DIR/backend/"
echo ""
echo "2. Copy your environment file:"
echo "   scp ./Model_Code/Keys_Security/env root@your-server:$APP_DIR/Keys_Security/env"
echo ""
echo "3. Set permissions on env file:"
echo "   sudo chmod 600 $APP_DIR/Keys_Security/env"
echo "   sudo chown $APP_USER:$APP_USER $APP_DIR/Keys_Security/env"
echo ""
echo "4. Start the services:"
echo "   sudo systemctl start vigil"
echo "   sudo systemctl restart nginx"
echo ""
echo "5. Setup SSL (after DNS is pointing to server):"
echo "   sudo certbot --nginx -d $DOMAIN -d www.$DOMAIN"
echo ""
echo "6. Check service status:"
echo "   sudo systemctl status vigil"
echo "   sudo systemctl status nginx"
echo ""
echo "7. View logs:"
echo "   sudo journalctl -u vigil -f"
echo "   sudo tail -f /var/log/nginx/vigil_error.log"
echo ""

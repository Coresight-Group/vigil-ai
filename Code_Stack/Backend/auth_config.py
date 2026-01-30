"""
Authentication Configuration for VIGIL
Managed by Coresight Group
"""

import os
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional

# ============================================
# EMAIL CONFIGURATION (TLS Encrypted)
# ============================================

class EmailConfig:
    """
    Secure email configuration for admin authentication
    Uses Transport Layer Security (TLS) encryption
    """

    # SMTP Configuration
    SMTP_HOST = "smtp.vigilsecure.com"  # Coresight Group mail server
    SMTP_PORT_PRIMARY = 465  # SSL/TLS
    SMTP_PORT_SECONDARY = 587  # STARTTLS
    SMTP_PORT_FALLBACK = 2525  # Alternative port

    # Email credentials (encrypted in production)
    SMTP_USERNAME = os.getenv('VIGIL_SMTP_USER', 'admin@vigilsecure.com')
    SMTP_PASSWORD = os.getenv('VIGIL_SMTP_PASS', '')  # Must be set in environment

    # TLS Configuration
    USE_TLS = True
    TLS_VERSION = 'TLSv1.3'  # Force TLS 1.3
    VERIFY_CERT = True

    @staticmethod
    def get_admin_email(client_id: str) -> str:
        """
        Generate or retrieve admin email for a specific client
        Format: admin@{random_string}{numbers}.com
        """
        # In production, this would retrieve from secure database
        return f"admin@{client_id}.vigilsecure.com"


# ============================================
# DNS CONFIGURATION
# ============================================

class DNSConfig:
    """
    DNS Records configuration for multi-tenant client access
    Each client gets unique subdomain and custom email domain
    """

    # Base domain
    BASE_DOMAIN = "vigilsecure.com"

    @staticmethod
    def generate_client_domain(business_name: str = None) -> Dict[str, str]:
        """
        Generate unique domain configuration for new client
        Returns DNS records needed for setup
        """
        # Generate random string for domain (12 chars: 8 letters + 4 numbers)
        random_letters = secrets.token_hex(4)  # 8 hex chars
        random_numbers = ''.join([str(secrets.randbelow(10)) for _ in range(4)])
        domain_id = f"{random_letters}{random_numbers}"

        client_subdomain = f"{domain_id}.vigilsecure.com"
        mail_domain = f"{domain_id}.vigilmail.com"

        return {
            "client_id": domain_id,
            "interface_domain": client_subdomain,
            "mail_domain": mail_domain,
            "admin_email": f"admin@{mail_domain}",
            "dns_records": {
                # A Record - Points to VIGIL interface server
                "A": {
                    "type": "A",
                    "name": client_subdomain,
                    "value": "203.0.113.10",  # Replace with actual server IP
                    "ttl": 3600
                },
                # CNAME Record - Interface alias
                "CNAME": {
                    "type": "CNAME",
                    "name": f"app.{client_subdomain}",
                    "value": client_subdomain,
                    "ttl": 3600
                },
                # MX Records - Mail exchange for admin email
                "MX_PRIMARY": {
                    "type": "MX",
                    "name": mail_domain,
                    "value": f"mail1.vigilsecure.com",
                    "priority": 10,
                    "ttl": 3600
                },
                "MX_SECONDARY": {
                    "type": "MX",
                    "name": mail_domain,
                    "value": f"mail2.vigilsecure.com",
                    "priority": 20,
                    "ttl": 3600
                },
                # SPF Record - Email authentication
                "TXT_SPF": {
                    "type": "TXT",
                    "name": mail_domain,
                    "value": "v=spf1 include:vigilsecure.com ~all",
                    "ttl": 3600
                },
                # DKIM Record - Email signing
                "TXT_DKIM": {
                    "type": "TXT",
                    "name": f"vigil._domainkey.{mail_domain}",
                    "value": "v=DKIM1; k=rsa; p=<PUBLIC_KEY_PLACEHOLDER>",
                    "ttl": 3600
                },
                # DMARC Record - Email policy
                "TXT_DMARC": {
                    "type": "TXT",
                    "name": f"_dmarc.{mail_domain}",
                    "value": "v=DMARC1; p=quarantine; rua=mailto:dmarc@vigilsecure.com",
                    "ttl": 3600
                }
            }
        }


# ============================================
# AUTHENTICATION CONFIGURATION
# ============================================

class AuthConfig:
    """
    Authentication settings and security parameters
    """

    # Token Configuration
    JWT_SECRET_KEY = os.getenv('VIGIL_JWT_SECRET', secrets.token_hex(32))
    JWT_ALGORITHM = 'HS256'
    ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours
    REFRESH_TOKEN_EXPIRE_DAYS = 30

    # 6-Digit Code Configuration
    AUTH_CODE_LENGTH = 6
    AUTH_CODE_EXPIRE_MINUTES = 10  # Code valid for 10 minutes
    MAX_AUTH_ATTEMPTS = 3  # Lock after 3 failed attempts

    # Password Requirements
    MIN_PASSWORD_LENGTH = 12
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_NUMBERS = True
    REQUIRE_SPECIAL_CHARS = True

    # Session Configuration
    SESSION_TIMEOUT_MINUTES = 30  # Inactive session timeout
    MAX_CONCURRENT_SESSIONS = 3  # Per user

    # Rate Limiting
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15


# ============================================
# USER ROLES AND PERMISSIONS
# ============================================

class RoleConfig:
    """
    Define user roles and their permissions
    """

    ROLES = {
        'admin': {
            'name': 'Administrator',
            'permissions': [
                'view_all_data',
                'manage_users',
                'view_admin_access',
                'delete_users',
                'view_user_activity',
                'modify_settings',
                'access_audit_logs',
                'reveal_credentials',
                'send_auth_codes'
            ],
            'require_2fa': True
        },
        'user': {
            'name': 'User',
            'permissions': [
                'view_own_data',
                'chat_interface',
                'search_documents',
                'view_alerts',
                'view_history'
            ],
            'require_2fa': False
        }
    }


# ============================================
# ADMIN VERIFICATION PROCESS
# ============================================

class AdminVerificationConfig:
    """
    First-time admin setup and verification
    Managed exclusively by Coresight Group backend
    """

    @staticmethod
    def create_admin_account(
        business_name: str,
        admin_name: str,
        admin_role: str,
        username: str,
        password_hash: str
    ) -> Dict:
        """
        Create new admin account (Backend only - Coresight Group)
        This is called during client onboarding

        Returns admin account details and DNS configuration
        """
        # Generate client domain and email
        dns_config = DNSConfig.generate_client_domain(business_name)

        admin_account = {
            'account_id': secrets.token_hex(16),
            'business_name': business_name,
            'admin_name': admin_name,
            'admin_role': admin_role,
            'username': username,
            'password_hash': password_hash,
            'admin_email': dns_config['admin_email'],
            'client_id': dns_config['client_id'],
            'interface_domain': dns_config['interface_domain'],
            'mail_domain': dns_config['mail_domain'],
            'role': 'admin',
            'status': 'pending_first_login',
            'created_at': datetime.utcnow().isoformat(),
            'created_by': 'coresight_group',
            'dns_records': dns_config['dns_records'],
            'requires_first_auth': True
        }

        return admin_account

    @staticmethod
    def complete_first_login(account_id: str, auth_code: str) -> bool:
        """
        Complete first-time admin login with 6-digit verification
        Updates account status to 'active'
        """
        # Verify auth code
        # Update account status
        # Enable full admin access
        pass


# ============================================
# UTILITY FUNCTIONS
# ============================================

def generate_auth_code() -> str:
    """Generate secure 6-digit authentication code"""
    return ''.join([str(secrets.randbelow(10)) for _ in range(6)])


def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt"""
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # iterations
    )
    return f"{salt}${pwd_hash.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    try:
        salt, pwd_hash = hashed.split('$')
        new_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return new_hash.hex() == pwd_hash
    except:
        return False


# ============================================
# EXAMPLE CLIENT SETUP
# ============================================

if __name__ == "__main__":
    # Example: Create new admin account for a client
    example_admin = AdminVerificationConfig.create_admin_account(
        business_name="Example Corp",
        admin_name="John Doe",
        admin_role="Chief Operations Officer",
        username="jdoe_admin",
        password_hash=hash_password("SecurePassword123!")
    )

    print("\n" + "="*60)
    print("NEW CLIENT SETUP - CORESIGHT GROUP INTERNAL")
    print("="*60)
    print(f"\nBusiness: {example_admin['business_name']}")
    print(f"Admin: {example_admin['admin_name']} ({example_admin['admin_role']})")
    print(f"Username: {example_admin['username']}")
    print(f"Client ID: {example_admin['client_id']}")
    print(f"\nInterface Domain: https://{example_admin['interface_domain']}")
    print(f"Admin Email: {example_admin['admin_email']}")
    print(f"\nDNS Records to Configure:")
    print("-" * 60)

    for record_name, record_data in example_admin['dns_records'].items():
        print(f"\n{record_name}:")
        for key, value in record_data.items():
            print(f"  {key}: {value}")

    print("\n" + "="*60)
    print("Account Status: PENDING FIRST LOGIN")
    print("Admin must complete 6-digit email verification on first login")
    print("="*60 + "\n")

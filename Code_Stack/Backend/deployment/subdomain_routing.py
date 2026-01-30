"""
Subdomain-Based Routing and Database Connection Manager
Routes clients to their isolated database/cache based on subdomain
"""

import os
from typing import Dict, Optional
import hashlib

# ============================================
# SUBDOMAIN ROUTING SYSTEM
# ============================================

class SubdomainRouter:
    """
    Routes incoming requests to correct database and cache
    based on subdomain (company name)

    Example Flow:
    1. Admin creates VIGIL account for "AcmeCorp"
    2. Coresight creates subdomain: acmecorp.vigilsecure.com
    3. DNS A record points to VIGIL server
    4. User visits acmecorp.vigilsecure.com
    5. Router extracts "acmecorp" from subdomain
    6. Routes to acmecorp_db database and acmecorp_cache
    7. Admin email: admin@acmecorp.vigilmail.com (same domain pattern)
    """

    def __init__(self):
        self.base_domain = "vigilsecure.com"
        self.mail_domain_suffix = "vigilmail.com"

    def extract_subdomain(self, hostname: str) -> str:
        """
        Extract subdomain (client identifier) from hostname

        Examples:
        - acmecorp.vigilsecure.com → acmecorp
        - techsolutions.vigilsecure.com → techsolutions
        - localhost → testclient (for development)
        """
        # Handle localhost/development
        if hostname in ['localhost', '127.0.0.1']:
            return 'testclient'

        # Extract subdomain (first part before base domain)
        parts = hostname.split('.')
        if len(parts) >= 3 and hostname.endswith(self.base_domain):
            return parts[0]

        # If can't determine, return default
        return 'default'

    def get_database_name(self, subdomain: str) -> str:
        """
        Get database name for subdomain
        Each client gets isolated database

        Format: {subdomain}_db
        Example: acmecorp → acmecorp_db
        """
        return f"{subdomain}_db"

    def get_cache_key_prefix(self, subdomain: str) -> str:
        """
        Get Redis cache key prefix for subdomain
        Ensures cache isolation between clients

        Format: vigil:{subdomain}:
        Example: vigil:acmecorp:session:12345
        """
        return f"vigil:{subdomain}:"

    def get_admin_email(self, subdomain: str) -> str:
        """
        Get admin email for subdomain
        Email domain matches subdomain pattern

        Format: admin@{subdomain}.vigilmail.com
        Example: acmecorp → admin@acmecorp.vigilmail.com

        Note: subdomain.vigilmail.com has same A records as subdomain.vigilsecure.com
        This ensures email domain and interface domain are linked
        """
        return f"admin@{subdomain}.{self.mail_domain_suffix}"

    def get_client_config(self, hostname: str) -> Dict[str, str]:
        """
        Get complete client configuration from hostname
        Returns all routing information needed
        """
        subdomain = self.extract_subdomain(hostname)

        return {
            'subdomain': subdomain,
            'database': self.get_database_name(subdomain),
            'cache_prefix': self.get_cache_key_prefix(subdomain),
            'admin_email': self.get_admin_email(subdomain),
            'interface_domain': f"{subdomain}.{self.base_domain}",
            'mail_domain': f"{subdomain}.{self.mail_domain_suffix}",
            'a_record': f"{subdomain}.{self.base_domain}",  # Points to server IP
            'mx_record': f"{subdomain}.{self.mail_domain_suffix}"  # Email routing
        }


# ============================================
# DATABASE CONNECTION MANAGER
# ============================================

class DatabaseConnectionManager:
    """
    Manages database connections per subdomain
    Each client gets isolated database instance
    """

    def __init__(self):
        self.connections = {}
        self.router = SubdomainRouter()

    def get_connection(self, subdomain: str):
        """
        Get or create database connection for subdomain
        Implements connection pooling
        """
        db_name = self.router.get_database_name(subdomain)

        if db_name not in self.connections:
            # Create new connection (pseudo-code)
            self.connections[db_name] = self._create_connection(db_name)

        return self.connections[db_name]

    def _create_connection(self, db_name: str):
        """
        Create database connection
        In production, use proper connection pooling (SQLAlchemy, etc.)
        """
        # Example using environment variables
        db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER', 'vigil_admin'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': db_name,
            'charset': 'utf8mb4',
            'autocommit': False
        }

        # Return connection object (pseudo-code)
        # In production: return mysql.connector.connect(**db_config)
        return db_config


# ============================================
# CACHE MANAGER (Redis)
# ============================================

class CacheManager:
    """
    Manages Redis cache with per-client key prefixes
    Ensures cache isolation between clients
    """

    def __init__(self):
        self.router = SubdomainRouter()
        self.redis_client = None  # Initialize Redis connection

    def get_key(self, subdomain: str, key: str) -> str:
        """
        Generate cache key with client prefix

        Example:
        subdomain="acmecorp", key="session:12345"
        → "vigil:acmecorp:session:12345"
        """
        prefix = self.router.get_cache_key_prefix(subdomain)
        return f"{prefix}{key}"

    def set(self, subdomain: str, key: str, value: str, ttl: int = 3600):
        """Set cache value with subdomain isolation"""
        full_key = self.get_key(subdomain, key)
        # self.redis_client.setex(full_key, ttl, value)
        pass

    def get(self, subdomain: str, key: str) -> Optional[str]:
        """Get cache value with subdomain isolation"""
        full_key = self.get_key(subdomain, key)
        # return self.redis_client.get(full_key)
        return None

    def delete(self, subdomain: str, key: str):
        """Delete cache value"""
        full_key = self.get_key(subdomain, key)
        # self.redis_client.delete(full_key)
        pass


# ============================================
# REQUEST MIDDLEWARE
# ============================================

class SubdomainMiddleware:
    """
    Middleware to extract subdomain and attach to request
    Use in Flask/FastAPI/Django
    """

    def __init__(self):
        self.router = SubdomainRouter()

    def process_request(self, request):
        """
        Process incoming request and attach client config

        Usage in Flask:
        @app.before_request
        def before_request():
            middleware = SubdomainMiddleware()
            middleware.process_request(request)

        Then access via: request.client_config
        """
        hostname = request.headers.get('Host', request.host)
        client_config = self.router.get_client_config(hostname)

        # Attach to request object
        request.client_config = client_config
        request.subdomain = client_config['subdomain']
        request.database = client_config['database']

        return client_config


# ============================================
# AUTHENTICATION CODE EMAIL SENDER
# ============================================

class AuthEmailSender:
    """
    Sends 6-digit authentication codes to admin emails
    Uses subdomain-based admin email addresses
    """

    def __init__(self):
        self.router = SubdomainRouter()

    def send_auth_code(self, subdomain: str, admin_username: str, code: str) -> bool:
        """
        Send authentication code to admin email

        Args:
            subdomain: Client subdomain (e.g., "acmecorp")
            admin_username: Admin username
            code: 6-digit authentication code

        Email sent to: admin@{subdomain}.vigilmail.com
        """
        admin_email = self.router.get_admin_email(subdomain)

        email_body = f"""
Hello {admin_username},

Your VIGIL authentication code is:

{code}

This code will expire in 10 minutes.

If you did not request this code, please contact Coresight Group immediately.

---
VIGIL by Coresight Group
Client: {subdomain}
"""

        # Send email via SMTP (implementation depends on email library)
        print(f"Sending auth code to: {admin_email}")
        print(f"Code: {code}")

        return True


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SUBDOMAIN ROUTING DEMONSTRATION")
    print("="*70 + "\n")

    router = SubdomainRouter()

    # Example 1: Acme Corp
    print("Example 1: Acme Corp")
    print("-" * 70)
    hostname1 = "acmecorp.vigilsecure.com"
    config1 = router.get_client_config(hostname1)

    print(f"Hostname: {hostname1}")
    print(f"  → Subdomain: {config1['subdomain']}")
    print(f"  → Database: {config1['database']}")
    print(f"  → Cache Prefix: {config1['cache_prefix']}")
    print(f"  → Admin Email: {config1['admin_email']}")
    print(f"  → A Record: {config1['a_record']} (points to VIGIL server)")
    print(f"  → MX Record: {config1['mail_domain']} (email routing)")
    print()

    # Example 2: Tech Solutions
    print("Example 2: Tech Solutions")
    print("-" * 70)
    hostname2 = "techsolutions.vigilsecure.com"
    config2 = router.get_client_config(hostname2)

    print(f"Hostname: {hostname2}")
    print(f"  → Subdomain: {config2['subdomain']}")
    print(f"  → Database: {config2['database']}")
    print(f"  → Cache Prefix: {config2['cache_prefix']}")
    print(f"  → Admin Email: {config2['admin_email']}")
    print(f"  → A Record: {config2['a_record']}")
    print(f"  → MX Record: {config2['mail_domain']}")
    print()

    # Example 3: Localhost (Development)
    print("Example 3: Localhost (Development)")
    print("-" * 70)
    hostname3 = "localhost"
    config3 = router.get_client_config(hostname3)

    print(f"Hostname: {hostname3}")
    print(f"  → Subdomain: {config3['subdomain']}")
    print(f"  → Database: {config3['database']}")
    print(f"  → Cache Prefix: {config3['cache_prefix']}")
    print(f"  → Admin Email: {config3['admin_email']}")
    print()

    # Show how subdomain matches email domain
    print("="*70)
    print("KEY CONCEPT: Subdomain = Email Domain Base")
    print("="*70)
    print("\nInterface Domain:  acmecorp.vigilsecure.com")
    print("Email Domain:      acmecorp.vigilmail.com")
    print("Admin Email:       admin@acmecorp.vigilmail.com")
    print("\n→ Both use 'acmecorp' as base identifier")
    print("→ A records point to same infrastructure")
    print("→ Links authentication to correct database/cache")
    print("="*70 + "\n")

    # Show cache key examples
    print("Cache Key Examples:")
    print("-" * 70)
    cache_mgr = CacheManager()
    print(f"Session: {cache_mgr.get_key('acmecorp', 'session:user123')}")
    print(f"Auth Code: {cache_mgr.get_key('acmecorp', 'auth:admin456')}")
    print(f"User Data: {cache_mgr.get_key('acmecorp', 'user:789:profile')}")
    print()

    # Show authentication flow
    print("="*70)
    print("AUTHENTICATION FLOW WITH SUBDOMAIN ROUTING")
    print("="*70)
    print("""
1. User visits: acmecorp.vigilsecure.com
2. Server extracts subdomain: "acmecorp"
3. Admin tries to login:
   - Validates against: acmecorp_db
   - Sends code to: admin@acmecorp.vigilmail.com
   - Stores auth token in: vigil:acmecorp:auth:token123
4. Admin enters 6-digit code
5. Verifies from: vigil:acmecorp:auth:token123
6. Creates session in: acmecorp_db.sessions
7. All queries route to: acmecorp_db

→ Complete isolation from other clients
→ Email domain matches interface domain
→ Database and cache automatically routed by subdomain
    """)
    print("="*70 + "\n")

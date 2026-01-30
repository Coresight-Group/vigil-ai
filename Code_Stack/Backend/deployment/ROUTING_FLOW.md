# VIGIL Subdomain Routing Flow
**How Users/Admins Connect to the Right Database**

---

## Overview

The subdomain in the URL is the **single source of truth** that determines:
- Which database to query
- Which cache to use
- Which admin email to send codes to

**There is NO email-based login** - emails are only used for sending authentication codes.

---

## Step-by-Step Flow

### **Scenario: Acme Corp Admin Login**

#### 1. User Navigates to URL
```
URL: https://acmecorp.vigilsecure.com
```

#### 2. Frontend Extracts Subdomain (script.js)
```javascript
function getSubdomain() {
    const hostname = window.location.hostname;  // "acmecorp.vigilsecure.com"
    const parts = hostname.split('.');          // ["acmecorp", "vigilsecure", "com"]
    return parts[0];                            // "acmecorp"
}

const CLIENT_SUBDOMAIN = getSubdomain();  // "acmecorp"
```

**Result:** Frontend knows client is "acmecorp"

---

#### 3. Admin Enters Credentials
```
Username: jane_admin
Password: SecurePassword123!
```

#### 4. Frontend Sends Login Request
```javascript
fetch('/api/auth/admin-login', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-Client-Subdomain': 'acmecorp'  // ← CRITICAL: Subdomain sent in header
    },
    body: JSON.stringify({
        username: 'jane_admin',
        password: 'SecurePassword123!',
        subdomain: 'acmecorp'  // ← Also sent in body
    })
})
```

**Key Point:** The subdomain is sent WITH the login request, so backend knows which database to check.

---

#### 5. Backend Receives Request (Express/Flask/FastAPI)

```python
# Example Flask middleware
@app.before_request
def route_to_correct_database():
    subdomain = request.headers.get('X-Client-Subdomain')  # "acmecorp"

    # Create router
    router = SubdomainRouter()
    config = router.get_client_config(subdomain)

    # config = {
    #     'subdomain': 'acmecorp',
    #     'database': 'acmecorp_db',              ← Database name
    #     'cache_prefix': 'vigil:acmecorp:',      ← Cache prefix
    #     'admin_email': 'admin@acmecorp.vigilmail.com',  ← Email address
    #     'interface_domain': 'acmecorp.vigilsecure.com',
    #     'mail_domain': 'acmecorp.vigilmail.com'
    # }

    # Attach to request for use in handlers
    request.client_config = config
```

**Result:** Backend now knows to query `acmecorp_db`

---

#### 6. Backend Queries Correct Database

```python
def admin_login(username, password, subdomain):
    # Get database connection for this subdomain
    db_manager = DatabaseConnectionManager()
    db = db_manager.get_connection(subdomain)  # Connects to "acmecorp_db"

    # Query admin from acmecorp_db.admins table
    query = """
        SELECT admin_id, username, password_hash, admin_email, admin_name
        FROM admins
        WHERE username = %s AND client_id = %s
    """
    result = db.execute(query, (username, subdomain))

    if result and verify_password(password, result['password_hash']):
        # Password correct!
        admin_email = result['admin_email']  # From database: "admin@acmecorp.vigilmail.com"

        # Generate 6-digit code
        auth_code = generate_auth_code()  # "482916"

        # Store in cache with subdomain prefix
        cache_manager = CacheManager()
        cache_key = f"vigil:acmecorp:auth:token123"
        cache_manager.set(subdomain, f"auth:token123", auth_code, ttl=600)

        # Send email to admin
        send_email(
            to=admin_email,  # "admin@acmecorp.vigilmail.com"
            subject="VIGIL Authentication Code",
            body=f"Your code is: {auth_code}"
        )

        return {
            'success': True,
            'authToken': 'token123',
            'message': 'Code sent to admin email'
        }
```

**Key Points:**
1. Database connection is to `acmecorp_db` (determined by subdomain)
2. Admin email is retrieved FROM that database (stored during admin account creation)
3. Code is cached with subdomain prefix: `vigil:acmecorp:auth:token123`

---

#### 7. Admin Receives Email

```
To: admin@acmecorp.vigilmail.com
Subject: VIGIL Authentication Code

Hello Jane Smith,

Your VIGIL authentication code is:

482916

This code will expire in 10 minutes.
```

---

#### 8. Admin Enters Code in Browser

Still on `https://acmecorp.vigilsecure.com`

```javascript
fetch('/api/auth/admin-verify-code', {
    method: 'POST',
    headers: {
        'X-Client-Subdomain': 'acmecorp'  // ← Still sending subdomain
    },
    body: JSON.stringify({
        authToken: 'token123',
        code: '482916',
        subdomain: 'acmecorp'
    })
})
```

---

#### 9. Backend Verifies Code

```python
def verify_admin_code(auth_token, code, subdomain):
    # Get cached code with subdomain prefix
    cache_manager = CacheManager()
    cached_code = cache_manager.get(subdomain, f"auth:{auth_token}")
    # Looks up: "vigil:acmecorp:auth:token123"

    if cached_code == code:
        # Code matches!
        # Create session in acmecorp_db
        db = db_manager.get_connection(subdomain)  # acmecorp_db

        session_token = generate_jwt_token(admin_id='admin123', subdomain='acmecorp')

        # Store session in acmecorp_db.sessions table
        db.execute("""
            INSERT INTO sessions (session_id, admin_id, token, expires_at)
            VALUES (%s, %s, %s, %s)
        """, (session_id, admin_id, session_token, expires_at))

        return {
            'success': True,
            'token': session_token,
            'username': 'jane_admin',
            'adminEmail': 'admin@acmecorp.vigilmail.com'
        }
```

---

## Key Architecture Points

### 1. **Subdomain is the Routing Key**
```
URL Subdomain → Database Name → Admin Email
acmecorp → acmecorp_db → admin@acmecorp.vigilmail.com
```

### 2. **Email Address Stored in Database**
When Coresight Group creates an admin account:

```python
# Backend script (run by Coresight Group only)
from auth_config import AdminVerificationConfig, hash_password

admin = AdminVerificationConfig.create_admin_account(
    business_name="Acme Corp",
    admin_name="Jane Smith",
    admin_role="CTO",
    username="jane_admin",
    password_hash=hash_password("TempPassword123!")
)

# admin = {
#     'client_id': 'acmecorp',
#     'admin_email': 'admin@acmecorp.vigilmail.com',  ← Generated and stored
#     'username': 'jane_admin',
#     ...
# }

# Insert into database
db.execute("""
    INSERT INTO clients (client_id, business_name, interface_domain, mail_domain)
    VALUES ('acmecorp', 'Acme Corp', 'acmecorp.vigilsecure.com', 'acmecorp.vigilmail.com')
""")

db.execute("""
    INSERT INTO admins (admin_id, client_id, username, password_hash, admin_email, admin_name)
    VALUES (UUID(), 'acmecorp', 'jane_admin', '<HASH>', 'admin@acmecorp.vigilmail.com', 'Jane Smith')
""")
```

### 3. **Database Isolation**
Each client has their own database:
```
acmecorp.vigilsecure.com → acmecorp_db
techsolutions.vigilsecure.com → techsolutions_db
globalcorp.vigilsecure.com → globalcorp_db
```

No cross-contamination possible.

### 4. **Cache Isolation**
Cache keys are prefixed with subdomain:
```
vigil:acmecorp:auth:token123
vigil:acmecorp:session:user456
vigil:techsolutions:auth:token789
vigil:techsolutions:session:user101
```

---

## How Users Connect to Right Database

### User Login Flow

1. **User visits**: `https://acmecorp.vigilsecure.com`
2. **User logs in**: Username "john_user", Password "UserPass123!"
3. **Frontend sends**:
   ```javascript
   fetch('/api/auth/user-login', {
       headers: { 'X-Client-Subdomain': 'acmecorp' },
       body: { username: 'john_user', password: 'UserPass123!', subdomain: 'acmecorp' }
   })
   ```
4. **Backend queries**: `acmecorp_db.users` table
   ```sql
   SELECT user_id, username, password_hash
   FROM users
   WHERE username = 'john_user' AND client_id = 'acmecorp'
   ```
5. **Session stored in**: `acmecorp_db.sessions` table

**Users never receive emails** - they just log in with username/password.

---

## How "Forgot Password" / "Create Account" Work

### Forgot Password Flow

1. **User fills form**:
   - Username: "john_user"
   - Admin Full Name: "Jane Smith"

2. **Frontend sends**:
   ```javascript
   fetch('/api/auth/forgot-password', {
       headers: { 'X-Client-Subdomain': 'acmecorp' },
       body: { username: 'john_user', adminName: 'Jane Smith', subdomain: 'acmecorp' }
   })
   ```

3. **Backend verifies**:
   ```python
   # Query acmecorp_db
   # 1. Check if username exists in acmecorp_db.users
   user = db.execute("SELECT * FROM users WHERE username = %s AND client_id = %s",
                     ('john_user', 'acmecorp'))

   # 2. Check if admin with that name exists in acmecorp_db.admins
   admin = db.execute("SELECT admin_email FROM admins WHERE admin_name = %s AND client_id = %s",
                      ('Jane Smith', 'acmecorp'))

   # 3. If both exist, send reset code to admin_email
   if user and admin:
       send_email(
           to=admin['admin_email'],  # "admin@acmecorp.vigilmail.com"
           subject="Password Reset Request",
           body=f"User {username} has requested a password reset. Code: {code}"
       )
   ```

---

## Security Features

### 1. **No Cross-Client Access**
Even if a user knows another client's subdomain, they can't access it because:
- Their username doesn't exist in that database
- Sessions are stored per-database
- Cache keys are prefixed with subdomain

### 2. **DNS A Records**
Each subdomain points to the SAME server IP:
```
acmecorp.vigilsecure.com → 203.0.113.10
techsolutions.vigilsecure.com → 203.0.113.10
globalcorp.vigilsecure.com → 203.0.113.10
```

Server handles routing internally based on subdomain.

### 3. **Email Domain Matching**
Interface domain and email domain use same base:
```
Interface: acmecorp.vigilsecure.com
Email:     admin@acmecorp.vigilmail.com
           └──────┘ Same subdomain base
```

This ensures email and interface are linked to same database.

---

## Summary

**The email doesn't "log in" or "know" anything.**

The flow is:
1. **URL subdomain** determines database
2. **Database** contains admin email
3. **Admin email** receives code
4. **Admin enters code** on same subdomain URL
5. **Backend verifies** using same database

Everything is tied to the subdomain in the URL that the user originally visited.

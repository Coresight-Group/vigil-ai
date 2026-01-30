# VIGIL Authentication System Setup Guide
**Coresight Group - Internal Documentation**

## Overview
This document outlines the setup process for VIGIL's authentication system, including admin account creation, DNS configuration, and email setup.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [New Client Onboarding](#new-client-onboarding)
3. [DNS Configuration](#dns-configuration)
4. [Email Server Setup](#email-server-setup)
5. [Admin Account Creation](#admin-account-creation)
6. [User Account Management](#user-account-management)
7. [Security Considerations](#security-considerations)

---

## Architecture Overview

### Multi-Tenant Structure
- Each client receives a unique subdomain and email domain
- Isolated data per client with shared infrastructure
- Role-based access control (Admin vs User)

### Authentication Flow
```
USER LOGIN:
1. Enter username + password
2. Backend validates credentials
3. JWT token issued
4. Redirect to interface

ADMIN LOGIN:
1. Enter username + password
2. Backend validates credentials
3. 6-digit code sent to admin@{client}.vigilmail.com
4. Admin enters code
5. Backend verifies code
6. JWT token issued with admin permissions
7. Redirect to interface with Admin Access tab
```

---

## New Client Onboarding

### Step 1: Client Information Gathering
Collect from client:
- Business Name
- Admin Name
- Admin Role/Title
- Desired Username
- Initial Password (will be hashed)

### Step 2: Run Backend Script
```python
python auth_config.py

# Or directly in Python:
from auth_config import AdminVerificationConfig, hash_password

admin_account = AdminVerificationConfig.create_admin_account(
    business_name="Acme Corporation",
    admin_name="Jane Smith",
    admin_role="Chief Technology Officer",
    username="jsmith_admin",
    password_hash=hash_password("InitialSecurePassword123!")
)

print(admin_account)
```

### Step 3: Record Output
The script generates:
- `client_id`: Unique identifier (e.g., "a3f5e8c912347890")
- `interface_domain`: Client's VIGIL access URL
- `mail_domain`: Email domain for admin communications
- `admin_email`: Encrypted admin email address
- `dns_records`: Complete DNS configuration

### Step 4: Database Entry
```sql
-- Insert client
INSERT INTO clients (client_id, business_name, interface_domain, mail_domain, status)
VALUES ('a3f5e8c912347890', 'Acme Corporation', 'a3f5e8c912347890.vigilsecure.com',
        'a3f5e8c912347890.vigilmail.com', 'pending');

-- Insert admin
INSERT INTO admins (admin_id, client_id, username, password_hash, admin_name, admin_role,
                    admin_email, status, requires_first_auth)
VALUES (UUID(), 'a3f5e8c912347890', 'jsmith_admin', '<HASHED_PASSWORD>',
        'Jane Smith', 'Chief Technology Officer', 'admin@a3f5e8c912347890.vigilmail.com',
        'pending_first_login', TRUE);
```

---

## DNS Configuration

### Required DNS Records

#### 1. A Record (Interface Access)
```
Type: A
Name: a3f5e8c912347890.vigilsecure.com
Value: 203.0.113.10  (VIGIL server IP)
TTL: 3600
```

#### 2. CNAME Record (App Alias)
```
Type: CNAME
Name: app.a3f5e8c912347890.vigilsecure.com
Value: a3f5e8c912347890.vigilsecure.com
TTL: 3600
```

#### 3. MX Records (Email)
```
Type: MX
Name: a3f5e8c912347890.vigilmail.com
Value: mail1.vigilsecure.com
Priority: 10
TTL: 3600

Type: MX
Name: a3f5e8c912347890.vigilmail.com
Value: mail2.vigilsecure.com
Priority: 20
TTL: 3600
```

#### 4. TXT Records (Email Authentication)

**SPF Record:**
```
Type: TXT
Name: a3f5e8c912347890.vigilmail.com
Value: v=spf1 include:vigilsecure.com ~all
TTL: 3600
```

**DKIM Record:**
```
Type: TXT
Name: vigil._domainkey.a3f5e8c912347890.vigilmail.com
Value: v=DKIM1; k=rsa; p=<PUBLIC_KEY>
TTL: 3600
```

**DMARC Record:**
```
Type: TXT
Name: _dmarc.a3f5e8c912347890.vigilmail.com
Value: v=DMARC1; p=quarantine; rua=mailto:dmarc@vigilsecure.com
TTL: 3600
```

### DNS Propagation
- Wait 24-48 hours for full DNS propagation
- Verify using: `nslookup a3f5e8c912347890.vigilsecure.com`

---

## Email Server Setup

### SMTP Configuration

**Primary Port (SSL/TLS):**
- Host: `smtp.vigilsecure.com`
- Port: `465`
- Encryption: SSL/TLS
- Protocol: SMTPS

**Secondary Port (STARTTLS):**
- Host: `smtp.vigilsecure.com`
- Port: `587`
- Encryption: STARTTLS
- Protocol: SMTP with TLS upgrade

**Fallback Port:**
- Host: `smtp.vigilsecure.com`
- Port: `2525`
- Encryption: TLS
- Use when 465/587 blocked

### TLS Configuration
```python
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

context = ssl.create_default_context()
context.check_hostname = True
context.verify_mode = ssl.CERT_REQUIRED
context.minimum_version = ssl.TLSVersion.TLSv1_3

# Connect via port 465 (SSL)
with smtplib.SMTP_SSL('smtp.vigilsecure.com', 465, context=context) as server:
    server.login('admin@vigilsecure.com', '<PASSWORD>')
    server.send_message(msg)
```

### Email Template for Auth Codes
```html
Subject: VIGIL Admin Authentication Code

Hello {admin_name},

Your 6-digit authentication code is:

{auth_code}

This code will expire in 10 minutes.

If you did not request this code, please contact Coresight Group immediately.

---
VIGIL by Coresight Group
Secure Supply Chain Intelligence
```

---

## Admin Account Creation

### Backend Process (Coresight Group Only)

**No frontend signup for admins.** All admin accounts created via backend:

```python
# Step 1: Generate admin account
from auth_config import AdminVerificationConfig, hash_password

admin = AdminVerificationConfig.create_admin_account(
    business_name="Example Corp",
    admin_name="John Doe",
    admin_role="COO",
    username="jdoe_admin",
    password_hash=hash_password("TempPassword123!")
)

# Step 2: Save to database
# (Use database_schema.sql structure)

# Step 3: Configure DNS
# (Apply dns_records from admin object)

# Step 4: Send credentials to client
# - Interface URL: https://{interface_domain}
# - Username: {username}
# - Temporary Password: (send securely)
# - Instruct to complete first login with 6-digit verification
```

### First Login Flow

1. Admin navigates to `https://{client_id}.vigilsecure.com`
2. Clicks "ADMIN LOGIN" on right panel
3. Enters username + temporary password
4. Backend validates and sends 6-digit code to `admin@{client_id}.vigilmail.com`
5. Admin checks email, enters code
6. Backend verifies code
7. Account status updated: `pending_first_login` → `active`
8. Admin prompted to change password
9. Access granted to interface with "Admin Access" tab in settings

---

## User Account Management

### Admin Creates User Accounts

In VIGIL interface, admin navigates to:
- Settings → Admin Access → "Create User"

**Process:**
1. Admin fills form:
   - Username
   - Email (optional)
   - Temporary password
2. Admin submits
3. Backend creates user account:
   ```sql
   INSERT INTO users (user_id, client_id, username, password_hash,
                      created_by_admin_id, status)
   VALUES (UUID(), '{client_id}', '{username}', '{hashed_password}',
           '{admin_id}', 'active');
   ```
4. User credentials sent to user (via admin or separate channel)
5. User logs in via "USER LOGIN" panel (left side)

### User Status Tracking

Admin can view all users in "Admin Access" tab:

| Username | Status | Last Login | Current State |
|----------|--------|------------|---------------|
| user1 | Active | 2 min ago | Logged In |
| user2 | Active | 15 min ago | Logged In (Inactive) |
| user3 | Active | 3 hours ago | Logged Out |
| user4 | Disabled | N/A | Disabled |

**States:**
- **Logged In**: Active session, recent activity (<5 min)
- **Logged In (Inactive)**: Active session, no activity (>5 min)
- **Logged Out**: No active session
- **Disabled**: Account disabled by admin

### Admin Actions on Users
- **View Details**: See user activity logs
- **Disable Account**: Prevent user login (reversible)
- **Delete Account**: Permanently remove user
- **Reset Password**: Generate new temporary password

---

## Security Considerations

### Password Requirements
- Minimum 12 characters
- Must contain:
  - Uppercase letter
  - Lowercase letter
  - Number
  - Special character

### Rate Limiting
- Max 5 login attempts per 15 minutes
- Account locked after 3 failed 6-digit code attempts
- Lockout duration: 15 minutes

### Session Management
- JWT tokens expire after 8 hours
- Refresh tokens valid for 30 days
- Max 3 concurrent sessions per user
- Inactive timeout: 30 minutes

### Audit Logging
All actions logged in `activity_logs` table:
- Login attempts (success/failure)
- User creation/deletion
- Settings changes
- Data access

### Data Isolation
- Each client's data completely isolated
- Cross-client access impossible
- Admin can only manage users in their client

### Encryption
- Passwords: PBKDF2-HMAC-SHA256 with 100,000 iterations
- JWT tokens: HS256 algorithm
- Email transport: TLS 1.3 only
- Database connections: TLS encrypted

---

## Troubleshooting

### Admin Can't Receive Auth Code
1. Check DNS MX records configured
2. Verify email server reachable
3. Check spam/junk folders
4. Regenerate code (max 3 attempts)

### User Login Fails
1. Verify account status (not disabled/locked)
2. Check password meets requirements
3. Review activity logs for lockout
4. Admin can reset password

### DNS Not Resolving
1. Wait 24-48 hours for propagation
2. Verify records with `dig` or `nslookup`
3. Check registrar DNS settings
4. Contact Coresight Group infrastructure team

---

## Support

**For Coresight Group Internal Use Only**

Infrastructure Team: infrastructure@coresightgroup.com
Security Team: security@coresightgroup.com
Documentation: docs.internal.coresightgroup.com

---

*Last Updated: 2024*
*Version: 1.0*

"""
VIGIL Data Sync Service - Automatic Data Retrieval and Storage

Automatically retrieves data from client external sources and stores it
in their Supabase databases using REST API.

SUPPORTED SOURCE TYPES:
- Google Drive (files, folders, documents)
- Webhooks (real-time push notifications)
- REST APIs (polling external endpoints)
- Databases (PostgreSQL, MySQL, MSSQL)
- SFTP/FTP (file transfers)
- File Uploads (direct file processing)

ARCHITECTURE:
External Source -> Connector -> Transformer -> Client Supabase DB
                       |
                   Sync Job Tracker (Master DB)
"""

import os
import json
import time
import hashlib
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import schedule

from supabase import create_client, Client
from dotenv import load_dotenv
from cryptography.fernet import Fernet

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# BASE CONNECTOR CLASS
# ============================================================================

class BaseConnector(ABC):
    """Abstract base class for all data source connectors."""

    def __init__(self, config: Dict, credentials: Dict = None):
        self.config = config
        self.credentials = credentials or {}
        self.last_sync_timestamp = None
        self.error_count = 0

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the data source."""
        pass

    @abstractmethod
    def fetch_data(self, since: datetime = None) -> List[Dict]:
        """Fetch data from the source, optionally since a timestamp."""
        pass

    @abstractmethod
    def health_check(self) -> Tuple[bool, str]:
        """Check if the connection is healthy."""
        pass

    def transform_record(self, record: Dict, field_mappings: Dict) -> Dict:
        """Transform a source record to target schema using field mappings."""
        if not field_mappings:
            return record

        transformed = {}
        for source_field, target_field in field_mappings.items():
            if source_field in record:
                # Handle nested fields (e.g., "metadata.author" -> "author")
                if '.' in target_field:
                    parts = target_field.split('.')
                    current = transformed
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = record[source_field]
                else:
                    transformed[target_field] = record[source_field]

        # Keep unmapped fields
        for key, value in record.items():
            if key not in field_mappings and key not in transformed:
                transformed[key] = value

        return transformed


# ============================================================================
# GOOGLE DRIVE CONNECTOR
# ============================================================================

class GoogleDriveConnector(BaseConnector):
    """
    Connector for Google Drive data sources.

    Supports:
    - Monitoring specific folders for new/changed files
    - Downloading documents (PDF, DOCX, XLSX, etc.)
    - Watching for changes via Drive API
    - Incremental sync based on modified time
    """

    GOOGLE_DRIVE_API = "https://www.googleapis.com/drive/v3"

    def __init__(self, config: Dict, credentials: Dict = None):
        super().__init__(config, credentials)
        self.access_token = credentials.get('access_token') if credentials else None
        self.refresh_token = credentials.get('refresh_token') if credentials else None
        self.folder_id = config.get('folder_id')
        self.file_types = config.get('file_types', ['pdf', 'docx', 'xlsx', 'csv'])

    def connect(self) -> bool:
        """Establish connection using OAuth token."""
        if not self.access_token:
            logger.error("Google Drive: No access token provided")
            return False

        try:
            # Test connection by listing files
            response = requests.get(
                f"{self.GOOGLE_DRIVE_API}/files",
                headers={"Authorization": f"Bearer {self.access_token}"},
                params={"pageSize": 1},
                timeout=10
            )

            if response.status_code == 401:
                # Token expired, try refresh
                if self._refresh_access_token():
                    return self.connect()
                return False

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Google Drive connection error: {e}")
            return False

    def _refresh_access_token(self) -> bool:
        """Refresh the OAuth access token."""
        if not self.refresh_token:
            return False

        client_id = self.credentials.get('client_id')
        client_secret = self.credentials.get('client_secret')

        if not client_id or not client_secret:
            return False

        try:
            response = requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": self.refresh_token,
                    "grant_type": "refresh_token"
                },
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get('access_token')
                return True

        except Exception as e:
            logger.error(f"Token refresh error: {e}")

        return False

    def fetch_data(self, since: datetime = None) -> List[Dict]:
        """Fetch files from Google Drive, optionally modified since timestamp."""
        if not self.connect():
            return []

        files = []
        page_token = None

        # Build query
        query_parts = []
        if self.folder_id:
            query_parts.append(f"'{self.folder_id}' in parents")
        query_parts.append("trashed = false")

        if since:
            # Format for Google API
            since_str = since.strftime("%Y-%m-%dT%H:%M:%S")
            query_parts.append(f"modifiedTime > '{since_str}'")

        query = " and ".join(query_parts)

        try:
            while True:
                params = {
                    "q": query,
                    "pageSize": 100,
                    "fields": "nextPageToken, files(id, name, mimeType, modifiedTime, size, webViewLink)",
                    "orderBy": "modifiedTime desc"
                }

                if page_token:
                    params["pageToken"] = page_token

                response = requests.get(
                    f"{self.GOOGLE_DRIVE_API}/files",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    params=params,
                    timeout=30
                )

                if response.status_code != 200:
                    logger.error(f"Google Drive API error: {response.text}")
                    break

                data = response.json()

                for file in data.get('files', []):
                    # Filter by file type
                    file_ext = file['name'].split('.')[-1].lower() if '.' in file['name'] else ''
                    if not self.file_types or file_ext in self.file_types:
                        files.append({
                            'source_id': file['id'],
                            'filename': file['name'],
                            'mime_type': file['mimeType'],
                            'modified_at': file['modifiedTime'],
                            'size': file.get('size', 0),
                            'web_link': file.get('webViewLink'),
                            'source_type': 'google_drive'
                        })

                page_token = data.get('nextPageToken')
                if not page_token:
                    break

        except Exception as e:
            logger.error(f"Google Drive fetch error: {e}")

        return files

    def download_file(self, file_id: str) -> Optional[bytes]:
        """Download file content from Google Drive."""
        if not self.connect():
            return None

        try:
            response = requests.get(
                f"{self.GOOGLE_DRIVE_API}/files/{file_id}",
                headers={"Authorization": f"Bearer {self.access_token}"},
                params={"alt": "media"},
                timeout=60
            )

            if response.status_code == 200:
                return response.content

        except Exception as e:
            logger.error(f"Google Drive download error: {e}")

        return None

    def health_check(self) -> Tuple[bool, str]:
        """Check Google Drive connection health."""
        try:
            if self.connect():
                return True, "Connected to Google Drive"
            return False, "Failed to authenticate with Google Drive"
        except Exception as e:
            return False, str(e)


# ============================================================================
# REST API CONNECTOR
# ============================================================================

class RestApiConnector(BaseConnector):
    """
    Connector for external REST APIs.

    Supports:
    - Multiple authentication methods (API key, OAuth2, Basic, Bearer)
    - Custom endpoints and methods
    - Rate limiting
    - Pagination handling
    """

    def __init__(self, config: Dict, credentials: Dict = None):
        super().__init__(config, credentials)
        self.base_url = config.get('base_url', '').rstrip('/')
        self.auth_type = config.get('auth_type', 'api_key')
        self.endpoints = config.get('endpoints', [])
        self.rate_limit_delay = config.get('rate_limit_delay', 0.5)
        self.headers = config.get('custom_headers', {})

    def connect(self) -> bool:
        """Test connection to the API."""
        try:
            headers = self._get_auth_headers()

            # Try a simple request to verify connection
            test_endpoint = self.endpoints[0] if self.endpoints else {'path': '/', 'method': 'GET'}

            response = requests.request(
                method=test_endpoint.get('method', 'GET'),
                url=f"{self.base_url}{test_endpoint.get('path', '/')}",
                headers=headers,
                timeout=10
            )

            return response.status_code in [200, 201, 204]

        except Exception as e:
            logger.error(f"REST API connection error: {e}")
            return False

    def _get_auth_headers(self) -> Dict:
        """Build authentication headers based on auth type."""
        headers = {**self.headers}

        if self.auth_type == 'api_key':
            api_key = self.credentials.get('api_key')
            key_header = self.config.get('api_key_header', 'X-API-Key')
            if api_key:
                headers[key_header] = api_key

        elif self.auth_type == 'bearer':
            token = self.credentials.get('token') or self.credentials.get('access_token')
            if token:
                headers['Authorization'] = f"Bearer {token}"

        elif self.auth_type == 'basic':
            import base64
            username = self.credentials.get('username', '')
            password = self.credentials.get('password', '')
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers['Authorization'] = f"Basic {credentials}"

        elif self.auth_type == 'oauth2':
            # OAuth2 token should be in credentials
            token = self.credentials.get('access_token')
            if token:
                headers['Authorization'] = f"Bearer {token}"

        return headers

    def fetch_data(self, since: datetime = None) -> List[Dict]:
        """Fetch data from all configured endpoints."""
        all_data = []
        headers = self._get_auth_headers()

        for endpoint in self.endpoints:
            endpoint_data = self._fetch_endpoint(endpoint, headers, since)
            all_data.extend(endpoint_data)

            # Rate limiting
            if self.rate_limit_delay:
                time.sleep(self.rate_limit_delay)

        return all_data

    def _fetch_endpoint(self, endpoint: Dict, headers: Dict, since: datetime = None) -> List[Dict]:
        """Fetch data from a single endpoint with pagination support."""
        records = []
        path = endpoint.get('path', '/')
        method = endpoint.get('method', 'GET')
        target_table = endpoint.get('map_to', 'documents')
        pagination = endpoint.get('pagination', {})

        # Build params
        params = endpoint.get('params', {}).copy()

        # Add timestamp filter if supported
        if since and endpoint.get('timestamp_param'):
            params[endpoint['timestamp_param']] = since.isoformat()

        page = 1
        has_more = True

        while has_more:
            try:
                # Handle pagination
                if pagination.get('type') == 'page':
                    params[pagination.get('page_param', 'page')] = page
                elif pagination.get('type') == 'offset':
                    params[pagination.get('offset_param', 'offset')] = (page - 1) * pagination.get('limit', 100)

                response = requests.request(
                    method=method,
                    url=f"{self.base_url}{path}",
                    headers=headers,
                    params=params if method == 'GET' else None,
                    json=params if method in ['POST', 'PUT'] else None,
                    timeout=30
                )

                if response.status_code not in [200, 201]:
                    logger.error(f"API error ({response.status_code}): {response.text[:200]}")
                    break

                data = response.json()

                # Extract records from response
                data_path = endpoint.get('data_path', '')
                if data_path:
                    for key in data_path.split('.'):
                        data = data.get(key, []) if isinstance(data, dict) else data

                if isinstance(data, list):
                    for record in data:
                        record['_source_table'] = target_table
                        record['_source_endpoint'] = path
                        records.append(record)
                elif isinstance(data, dict):
                    data['_source_table'] = target_table
                    data['_source_endpoint'] = path
                    records.append(data)

                # Check for more pages
                if pagination.get('type'):
                    page_size = len(data) if isinstance(data, list) else 1
                    has_more = page_size >= pagination.get('limit', 100)
                    page += 1
                else:
                    has_more = False

            except Exception as e:
                logger.error(f"Endpoint fetch error ({path}): {e}")
                has_more = False

        return records

    def health_check(self) -> Tuple[bool, str]:
        """Check REST API connection health."""
        try:
            if self.connect():
                return True, f"Connected to {self.base_url}"
            return False, f"Failed to connect to {self.base_url}"
        except Exception as e:
            return False, str(e)


# ============================================================================
# WEBHOOK RECEIVER
# ============================================================================

class WebhookReceiver:
    """
    Receives and processes webhook payloads from external sources.

    Webhooks are processed asynchronously and stored in a queue
    for the sync service to process.
    """

    def __init__(self, secret: str = None):
        self.secret = secret
        self.payload_queue = Queue()
        self.processed_count = 0

    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature using HMAC."""
        if not self.secret:
            return True  # No secret configured, skip verification

        import hmac
        expected = hmac.new(
            self.secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()

        # Compare signatures (constant-time comparison)
        return hmac.compare_digest(f"sha256={expected}", signature)

    def receive(self, payload: Dict, source: str, signature: str = None) -> bool:
        """Receive and queue a webhook payload."""
        try:
            # Verify signature if provided
            if signature and not self.verify_signature(json.dumps(payload).encode(), signature):
                logger.warning(f"Invalid webhook signature from {source}")
                return False

            # Add metadata
            enriched_payload = {
                'source': source,
                'received_at': datetime.now().isoformat(),
                'payload': payload,
                'source_type': 'webhook'
            }

            self.payload_queue.put(enriched_payload)
            self.processed_count += 1

            logger.info(f"Webhook received from {source}")
            return True

        except Exception as e:
            logger.error(f"Webhook receive error: {e}")
            return False

    def get_pending(self) -> List[Dict]:
        """Get all pending webhook payloads."""
        payloads = []
        while not self.payload_queue.empty():
            try:
                payloads.append(self.payload_queue.get_nowait())
            except:
                break
        return payloads


# ============================================================================
# DATABASE CONNECTOR
# ============================================================================

class DatabaseConnector(BaseConnector):
    """
    Connector for external databases.

    Supports:
    - PostgreSQL
    - MySQL
    - MSSQL

    Uses incremental sync based on timestamp columns.
    """

    def __init__(self, config: Dict, credentials: Dict = None):
        super().__init__(config, credentials)
        self.db_type = config.get('db_type', 'postgresql')
        self.host = config.get('host')
        self.port = config.get('port')
        self.database = config.get('database')
        self.tables = config.get('tables', [])
        self.connection = None

    def connect(self) -> bool:
        """Establish database connection."""
        try:
            username = self.credentials.get('username')
            password = self.credentials.get('password')

            if self.db_type == 'postgresql':
                import psycopg2
                self.connection = psycopg2.connect(
                    host=self.host,
                    port=self.port or 5432,
                    database=self.database,
                    user=username,
                    password=password,
                    connect_timeout=10
                )
            elif self.db_type == 'mysql':
                import mysql.connector
                self.connection = mysql.connector.connect(
                    host=self.host,
                    port=self.port or 3306,
                    database=self.database,
                    user=username,
                    password=password,
                    connect_timeout=10
                )
            elif self.db_type == 'mssql':
                import pyodbc
                conn_str = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={self.host},{self.port or 1433};"
                    f"DATABASE={self.database};"
                    f"UID={username};"
                    f"PWD={password};"
                    f"Connection Timeout=10"
                )
                self.connection = pyodbc.connect(conn_str)
            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                return False

            return True

        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return False

    def fetch_data(self, since: datetime = None) -> List[Dict]:
        """Fetch data from configured tables."""
        if not self.connect():
            return []

        all_records = []

        try:
            cursor = self.connection.cursor()

            for table_config in self.tables:
                source_table = table_config.get('source')
                target_table = table_config.get('target')
                timestamp_column = table_config.get('timestamp_column', 'updated_at')

                # Build query
                query = f"SELECT * FROM {source_table}"
                params = []

                if since and timestamp_column:
                    query += f" WHERE {timestamp_column} > %s"
                    params.append(since)

                query += f" ORDER BY {timestamp_column} DESC LIMIT 1000"

                cursor.execute(query, params)
                columns = [desc[0] for desc in cursor.description]

                for row in cursor.fetchall():
                    record = dict(zip(columns, row))
                    record['_source_table'] = target_table
                    record['_db_source'] = source_table

                    # Convert datetime objects to ISO strings
                    for key, value in record.items():
                        if isinstance(value, datetime):
                            record[key] = value.isoformat()

                    all_records.append(record)

        except Exception as e:
            logger.error(f"Database fetch error: {e}")
        finally:
            if self.connection:
                self.connection.close()
                self.connection = None

        return all_records

    def health_check(self) -> Tuple[bool, str]:
        """Check database connection health."""
        try:
            if self.connect():
                self.connection.close()
                self.connection = None
                return True, f"Connected to {self.db_type}://{self.host}/{self.database}"
            return False, f"Failed to connect to {self.db_type}://{self.host}/{self.database}"
        except Exception as e:
            return False, str(e)


# ============================================================================
# SFTP/FTP CONNECTOR
# ============================================================================

class SFTPConnector(BaseConnector):
    """
    Connector for SFTP/FTP file transfers.

    Monitors remote directories for new/changed files and downloads them.
    """

    def __init__(self, config: Dict, credentials: Dict = None):
        super().__init__(config, credentials)
        self.host = config.get('host')
        self.port = config.get('port', 22)
        self.remote_path = config.get('remote_path', '/')
        self.file_patterns = config.get('file_patterns', ['*'])
        self.protocol = config.get('protocol', 'sftp')  # 'sftp' or 'ftp'
        self.connection = None

    def connect(self) -> bool:
        """Establish SFTP/FTP connection."""
        try:
            username = self.credentials.get('username')
            password = self.credentials.get('password')
            private_key = self.credentials.get('private_key')

            if self.protocol == 'sftp':
                import paramiko
                transport = paramiko.Transport((self.host, self.port))

                if private_key:
                    # Use private key authentication
                    key = paramiko.RSAKey.from_private_key_file(private_key)
                    transport.connect(username=username, pkey=key)
                else:
                    transport.connect(username=username, password=password)

                self.connection = paramiko.SFTPClient.from_transport(transport)

            elif self.protocol == 'ftp':
                from ftplib import FTP, FTP_TLS
                use_tls = self.config.get('use_tls', False)

                if use_tls:
                    self.connection = FTP_TLS(self.host)
                    self.connection.login(username, password)
                    self.connection.prot_p()
                else:
                    self.connection = FTP(self.host)
                    self.connection.login(username, password)

            return True

        except Exception as e:
            logger.error(f"SFTP/FTP connection error: {e}")
            return False

    def fetch_data(self, since: datetime = None) -> List[Dict]:
        """Fetch file metadata from remote directory."""
        if not self.connect():
            return []

        files = []

        try:
            if self.protocol == 'sftp':
                # List files in remote directory
                for entry in self.connection.listdir_attr(self.remote_path):
                    # Check if file matches patterns
                    import fnmatch
                    matches = any(fnmatch.fnmatch(entry.filename, p) for p in self.file_patterns)

                    if matches and not entry.filename.startswith('.'):
                        mtime = datetime.fromtimestamp(entry.st_mtime)

                        # Skip if older than since
                        if since and mtime <= since:
                            continue

                        files.append({
                            'source_id': f"{self.host}:{self.remote_path}/{entry.filename}",
                            'filename': entry.filename,
                            'remote_path': f"{self.remote_path}/{entry.filename}",
                            'size': entry.st_size,
                            'modified_at': mtime.isoformat(),
                            'source_type': 'sftp'
                        })

            elif self.protocol == 'ftp':
                # FTP listing
                self.connection.cwd(self.remote_path)
                file_list = []
                self.connection.retrlines('MLSD', file_list.append)

                for line in file_list:
                    parts = line.split(';')
                    facts = {}
                    for part in parts[:-1]:
                        key, value = part.split('=')
                        facts[key.lower()] = value

                    filename = parts[-1].strip()

                    if facts.get('type') == 'file':
                        files.append({
                            'source_id': f"{self.host}:{self.remote_path}/{filename}",
                            'filename': filename,
                            'remote_path': f"{self.remote_path}/{filename}",
                            'size': int(facts.get('size', 0)),
                            'modified_at': facts.get('modify', ''),
                            'source_type': 'ftp'
                        })

        except Exception as e:
            logger.error(f"SFTP/FTP fetch error: {e}")
        finally:
            self._close()

        return files

    def download_file(self, remote_path: str) -> Optional[bytes]:
        """Download file content from remote server."""
        if not self.connect():
            return None

        try:
            import io
            buffer = io.BytesIO()

            if self.protocol == 'sftp':
                self.connection.getfo(remote_path, buffer)
            else:
                self.connection.retrbinary(f'RETR {remote_path}', buffer.write)

            buffer.seek(0)
            return buffer.read()

        except Exception as e:
            logger.error(f"SFTP/FTP download error: {e}")
            return None
        finally:
            self._close()

    def _close(self):
        """Close connection."""
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
            self.connection = None

    def health_check(self) -> Tuple[bool, str]:
        """Check SFTP/FTP connection health."""
        try:
            if self.connect():
                self._close()
                return True, f"Connected to {self.protocol}://{self.host}"
            return False, f"Failed to connect to {self.protocol}://{self.host}"
        except Exception as e:
            return False, str(e)


# ============================================================================
# DROPBOX CONNECTOR
# ============================================================================

class DropboxConnector(BaseConnector):
    """
    Connector for Dropbox data sources.

    Monitors folders for new/changed files using Dropbox API.
    """

    DROPBOX_API = "https://api.dropboxapi.com/2"
    DROPBOX_CONTENT = "https://content.dropboxapi.com/2"

    def __init__(self, config: Dict, credentials: Dict = None):
        super().__init__(config, credentials)
        self.access_token = credentials.get('access_token') if credentials else None
        self.folder_path = config.get('folder_path', '')
        self.file_types = config.get('file_types', [])

    def connect(self) -> bool:
        """Verify Dropbox connection."""
        if not self.access_token:
            return False

        try:
            response = requests.post(
                f"{self.DROPBOX_API}/users/get_current_account",
                headers={"Authorization": f"Bearer {self.access_token}"},
                timeout=10
            )
            return response.status_code == 200
        except:
            return False

    def fetch_data(self, since: datetime = None) -> List[Dict]:
        """Fetch files from Dropbox folder."""
        if not self.connect():
            return []

        files = []

        try:
            response = requests.post(
                f"{self.DROPBOX_API}/files/list_folder",
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "path": self.folder_path or "",
                    "recursive": True,
                    "include_deleted": False
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                for entry in data.get('entries', []):
                    if entry.get('.tag') == 'file':
                        # Filter by file type
                        ext = entry['name'].split('.')[-1].lower() if '.' in entry['name'] else ''
                        if self.file_types and ext not in self.file_types:
                            continue

                        # Check modified time
                        modified = entry.get('server_modified', '')
                        if since and modified:
                            mod_time = datetime.fromisoformat(modified.replace('Z', '+00:00'))
                            if mod_time <= since:
                                continue

                        files.append({
                            'source_id': entry['id'],
                            'filename': entry['name'],
                            'path': entry['path_display'],
                            'size': entry.get('size', 0),
                            'modified_at': modified,
                            'content_hash': entry.get('content_hash'),
                            'source_type': 'dropbox'
                        })

        except Exception as e:
            logger.error(f"Dropbox fetch error: {e}")

        return files

    def health_check(self) -> Tuple[bool, str]:
        if self.connect():
            return True, "Connected to Dropbox"
        return False, "Failed to connect to Dropbox"


# ============================================================================
# SHAREPOINT/ONEDRIVE CONNECTOR
# ============================================================================

class SharePointConnector(BaseConnector):
    """
    Connector for SharePoint/OneDrive data sources.

    Uses Microsoft Graph API to access files.
    """

    GRAPH_API = "https://graph.microsoft.com/v1.0"

    def __init__(self, config: Dict, credentials: Dict = None):
        super().__init__(config, credentials)
        self.access_token = credentials.get('access_token') if credentials else None
        self.site_id = config.get('site_id')
        self.drive_id = config.get('drive_id')
        self.folder_path = config.get('folder_path', '/root')

    def connect(self) -> bool:
        """Verify Microsoft Graph connection."""
        if not self.access_token:
            return False

        try:
            response = requests.get(
                f"{self.GRAPH_API}/me",
                headers={"Authorization": f"Bearer {self.access_token}"},
                timeout=10
            )
            return response.status_code == 200
        except:
            return False

    def fetch_data(self, since: datetime = None) -> List[Dict]:
        """Fetch files from SharePoint/OneDrive."""
        if not self.connect():
            return []

        files = []

        try:
            # Build endpoint based on configuration
            if self.site_id and self.drive_id:
                endpoint = f"{self.GRAPH_API}/sites/{self.site_id}/drives/{self.drive_id}/root/children"
            elif self.drive_id:
                endpoint = f"{self.GRAPH_API}/drives/{self.drive_id}/root/children"
            else:
                endpoint = f"{self.GRAPH_API}/me/drive/root/children"

            response = requests.get(
                endpoint,
                headers={"Authorization": f"Bearer {self.access_token}"},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                for item in data.get('value', []):
                    if 'file' in item:  # It's a file, not a folder
                        modified = item.get('lastModifiedDateTime', '')

                        if since and modified:
                            mod_time = datetime.fromisoformat(modified.replace('Z', '+00:00'))
                            if mod_time <= since:
                                continue

                        files.append({
                            'source_id': item['id'],
                            'filename': item['name'],
                            'web_url': item.get('webUrl'),
                            'size': item.get('size', 0),
                            'modified_at': modified,
                            'mime_type': item.get('file', {}).get('mimeType'),
                            'source_type': 'sharepoint'
                        })

        except Exception as e:
            logger.error(f"SharePoint fetch error: {e}")

        return files

    def health_check(self) -> Tuple[bool, str]:
        if self.connect():
            return True, "Connected to SharePoint/OneDrive"
        return False, "Failed to connect to SharePoint/OneDrive"


# ============================================================================
# FILE UPLOAD PROCESSOR
# ============================================================================

class FileUploadProcessor(BaseConnector):
    """
    Processes files uploaded directly to the system.

    Monitors a Supabase Storage bucket for new uploads and processes them.
    """

    def __init__(self, config: Dict, credentials: Dict = None):
        super().__init__(config, credentials)
        self.bucket = config.get('bucket', 'uploads')
        self.allowed_types = config.get('allowed_types', ['pdf', 'docx', 'xlsx', 'csv', 'txt', 'json'])
        self.auto_process = config.get('auto_process', True)
        self.supabase_url = config.get('supabase_url')
        self.supabase_key = config.get('supabase_key')
        self.client = None

    def connect(self) -> bool:
        """Connect to Supabase Storage."""
        if not self.supabase_url or not self.supabase_key:
            return False

        try:
            self.client = create_client(self.supabase_url, self.supabase_key)
            # Test by listing buckets
            self.client.storage.list_buckets()
            return True
        except Exception as e:
            logger.error(f"Supabase Storage connection error: {e}")
            return False

    def fetch_data(self, since: datetime = None) -> List[Dict]:
        """List files in the upload bucket."""
        if not self.connect():
            return []

        files = []

        try:
            # List files in bucket
            result = self.client.storage.from_(self.bucket).list()

            for item in result:
                if item.get('name') and not item['name'].endswith('/'):
                    # Filter by file type
                    ext = item['name'].split('.')[-1].lower() if '.' in item['name'] else ''
                    if self.allowed_types and ext not in self.allowed_types:
                        continue

                    files.append({
                        'source_id': item['id'],
                        'filename': item['name'],
                        'storage_path': f"{self.bucket}/{item['name']}",
                        'size': item.get('metadata', {}).get('size', 0),
                        'mime_type': item.get('metadata', {}).get('mimetype'),
                        'created_at': item.get('created_at'),
                        'source_type': 'file_upload'
                    })

        except Exception as e:
            logger.error(f"File upload fetch error: {e}")

        return files

    def health_check(self) -> Tuple[bool, str]:
        if self.connect():
            return True, f"Connected to storage bucket: {self.bucket}"
        return False, f"Failed to connect to storage bucket: {self.bucket}"


# ============================================================================
# SYNC SERVICE
# ============================================================================

class DataSyncService:
    """
    Main service for automatic data synchronization.

    - Manages connectors for each data source
    - Schedules sync jobs based on frequency
    - Transforms data to target schema
    - Stores data in client Supabase databases
    - Tracks sync job status and errors
    """

    def __init__(self):
        # Master database connection
        self.master_url = os.getenv('MASTER_SUPABASE_URL')
        self.master_key = os.getenv('MASTER_SUPABASE_SERVICE_KEY')
        self.encryption_key = os.getenv('ENCRYPTION_KEY')

        if self.master_url and self.master_key:
            self.master_client = create_client(self.master_url, self.master_key)
        else:
            self.master_client = None
            logger.warning("Master database not configured")

        # Encryption
        if self.encryption_key:
            self.fernet = Fernet(self.encryption_key.encode())
        else:
            self.fernet = None

        # Connector instances
        self.connectors: Dict[str, BaseConnector] = {}

        # Webhook receiver
        self.webhook_receiver = WebhookReceiver()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)

        # Running flag
        self.running = False

    def get_connector(self, source_type: str, config: Dict, credentials: Dict = None) -> Optional[BaseConnector]:
        """Factory method to create appropriate connector for ALL source types."""
        connectors = {
            # Cloud Storage
            'google_drive': GoogleDriveConnector,
            'dropbox': DropboxConnector,
            'sharepoint': SharePointConnector,
            'onedrive': SharePointConnector,  # Same connector, different config

            # REST APIs
            'api': RestApiConnector,
            'rest_api': RestApiConnector,

            # Databases
            'database': DatabaseConnector,
            'postgresql': DatabaseConnector,
            'mysql': DatabaseConnector,
            'mssql': DatabaseConnector,

            # File Transfer
            'sftp': SFTPConnector,
            'ftp': SFTPConnector,

            # Direct Uploads
            'file_upload': FileUploadProcessor,
            'storage': FileUploadProcessor,
        }

        connector_class = connectors.get(source_type)
        if connector_class:
            return connector_class(config, credentials)

        logger.warning(f"Unknown source type: {source_type}")
        return None

    def decrypt_credentials(self, encrypted: str) -> Dict:
        """Decrypt stored credentials."""
        if not encrypted or not self.fernet:
            return {}

        try:
            decrypted = self.fernet.decrypt(encrypted.encode()).decode()
            return json.loads(decrypted)
        except Exception as e:
            logger.error(f"Credential decryption error: {e}")
            return {}

    def get_pending_syncs(self) -> List[Dict]:
        """Get all data sources due for synchronization."""
        if not self.master_client:
            return []

        try:
            result = self.master_client.rpc('get_pending_syncs').execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting pending syncs: {e}")
            return []

    def get_client_connection(self, client_id: str) -> Optional[Tuple[Client, Dict]]:
        """Get Supabase client connection for a specific client."""
        if not self.master_client:
            return None

        try:
            result = self.master_client.table('clients').select(
                'id, name, slug, supabase_url, supabase_anon_key, supabase_service_key, settings'
            ).eq('id', client_id).eq('status', 'active').single().execute()

            if result.data:
                client_info = result.data

                # Decrypt service key
                service_key = client_info.get('supabase_service_key')
                if service_key and self.fernet:
                    try:
                        service_key = self.fernet.decrypt(service_key.encode()).decode()
                    except:
                        pass

                # Create connection
                client = create_client(
                    client_info['supabase_url'],
                    service_key or client_info['supabase_anon_key']
                )

                return client, client_info

        except Exception as e:
            logger.error(f"Error getting client connection: {e}")

        return None

    def create_sync_job(self, client_id: str, data_source_id: str, job_type: str = 'incremental') -> Optional[str]:
        """Create a new sync job record."""
        if not self.master_client:
            return None

        try:
            result = self.master_client.table('sync_jobs').insert({
                'client_id': client_id,
                'data_source_id': data_source_id,
                'job_type': job_type,
                'status': 'running',
                'started_at': datetime.now().isoformat(),
                'progress_percent': 0,
                'current_step': 'Initializing'
            }).execute()

            if result.data:
                return result.data[0]['id']

        except Exception as e:
            logger.error(f"Error creating sync job: {e}")

        return None

    def update_sync_job(self, job_id: str, updates: Dict):
        """Update sync job status and progress."""
        if not self.master_client:
            return

        try:
            self.master_client.table('sync_jobs').update(updates).eq('id', job_id).execute()
        except Exception as e:
            logger.error(f"Error updating sync job: {e}")

    def complete_sync_job(
        self,
        job_id: str,
        success: bool,
        records_processed: int = 0,
        records_created: int = 0,
        records_updated: int = 0,
        records_failed: int = 0,
        error_message: str = None
    ):
        """Mark sync job as completed."""
        if not self.master_client:
            return

        try:
            started_at = self.master_client.table('sync_jobs').select('started_at').eq('id', job_id).single().execute()

            duration_ms = 0
            if started_at.data:
                start_time = datetime.fromisoformat(started_at.data['started_at'].replace('Z', '+00:00'))
                duration_ms = int((datetime.now(start_time.tzinfo) - start_time).total_seconds() * 1000)

            self.master_client.table('sync_jobs').update({
                'status': 'completed' if success else 'failed',
                'completed_at': datetime.now().isoformat(),
                'duration_ms': duration_ms,
                'records_processed': records_processed,
                'records_created': records_created,
                'records_updated': records_updated,
                'records_failed': records_failed,
                'progress_percent': 100 if success else 0,
                'current_step': 'Completed' if success else 'Failed',
                'error_message': error_message
            }).eq('id', job_id).execute()

        except Exception as e:
            logger.error(f"Error completing sync job: {e}")

    def update_data_source(self, data_source_id: str, updates: Dict):
        """Update data source status and timestamps."""
        if not self.master_client:
            return

        try:
            self.master_client.table('data_sources').update(updates).eq('id', data_source_id).execute()
        except Exception as e:
            logger.error(f"Error updating data source: {e}")

    def sync_data_source(self, data_source: Dict) -> Dict:
        """
        Synchronize a single data source.

        Returns sync result with statistics.
        """
        data_source_id = data_source['data_source_id']
        client_id = data_source['client_id']
        source_type = data_source['source_type']
        config = data_source.get('connection_config', {})

        logger.info(f"Starting sync for data source {data_source_id} (type: {source_type})")

        # Create sync job
        job_id = self.create_sync_job(client_id, data_source_id)

        result = {
            'success': False,
            'records_processed': 0,
            'records_created': 0,
            'records_updated': 0,
            'records_failed': 0,
            'error': None
        }

        try:
            # Update data source status
            self.update_data_source(data_source_id, {
                'status': 'syncing',
                'error_message': None
            })

            # Get full data source details
            if self.master_client:
                ds_result = self.master_client.table('data_sources').select('*').eq('id', data_source_id).single().execute()
                if ds_result.data:
                    full_config = ds_result.data.get('connection_config', config)
                    encrypted_creds = ds_result.data.get('credentials_encrypted')
                    field_mappings = ds_result.data.get('field_mappings', {})
                    last_sync = ds_result.data.get('last_sync_at')
                else:
                    full_config = config
                    encrypted_creds = None
                    field_mappings = {}
                    last_sync = None
            else:
                full_config = config
                encrypted_creds = None
                field_mappings = {}
                last_sync = None

            # Decrypt credentials
            credentials = self.decrypt_credentials(encrypted_creds) if encrypted_creds else {}

            # Get connector
            connector = self.get_connector(source_type, full_config, credentials)
            if not connector:
                raise Exception(f"No connector available for source type: {source_type}")

            # Health check
            self.update_sync_job(job_id, {'current_step': 'Connecting to source'})
            healthy, health_msg = connector.health_check()
            if not healthy:
                raise Exception(f"Health check failed: {health_msg}")

            # Fetch data
            self.update_sync_job(job_id, {'current_step': 'Fetching data', 'progress_percent': 20})
            since = datetime.fromisoformat(last_sync.replace('Z', '+00:00')) if last_sync else None
            records = connector.fetch_data(since)

            result['records_processed'] = len(records)
            logger.info(f"Fetched {len(records)} records from source")

            if not records:
                # No new data
                result['success'] = True
                self.complete_sync_job(job_id, True, 0, 0, 0, 0)
                self.update_data_source(data_source_id, {
                    'status': 'active',
                    'last_sync_at': datetime.now().isoformat(),
                    'next_sync_at': self._calculate_next_sync(ds_result.data if self.master_client else {}),
                    'health_status': 'healthy',
                    'last_health_check': datetime.now().isoformat()
                })
                return result

            # Get client connection
            self.update_sync_job(job_id, {'current_step': 'Connecting to client database', 'progress_percent': 40})
            client_conn = self.get_client_connection(client_id)
            if not client_conn:
                raise Exception("Failed to connect to client database")

            client_db, client_info = client_conn

            # Process and store records
            self.update_sync_job(job_id, {'current_step': 'Storing data', 'progress_percent': 60})

            for i, record in enumerate(records):
                try:
                    # Determine target table
                    target_table = record.pop('_source_table', 'documents')
                    record.pop('_source_endpoint', None)
                    record.pop('_db_source', None)

                    # Apply field mappings
                    table_mappings = field_mappings.get(target_table, {}).get('source_fields', {})
                    transformed = connector.transform_record(record, table_mappings)

                    # Add metadata
                    transformed['synced_at'] = datetime.now().isoformat()
                    transformed['sync_source'] = source_type

                    # Check if record exists (upsert logic)
                    source_id = transformed.get('source_id') or transformed.get('id')
                    if source_id:
                        existing = client_db.table(target_table).select('id').eq('source_id', source_id).execute()

                        if existing.data:
                            # Update existing
                            client_db.table(target_table).update(transformed).eq('source_id', source_id).execute()
                            result['records_updated'] += 1
                        else:
                            # Insert new
                            transformed['source_id'] = source_id
                            client_db.table(target_table).insert(transformed).execute()
                            result['records_created'] += 1
                    else:
                        # Insert without upsert
                        client_db.table(target_table).insert(transformed).execute()
                        result['records_created'] += 1

                    # Update progress
                    progress = 60 + int((i + 1) / len(records) * 35)
                    if i % 10 == 0:
                        self.update_sync_job(job_id, {'progress_percent': progress})

                except Exception as e:
                    logger.error(f"Error storing record: {e}")
                    result['records_failed'] += 1

            result['success'] = True

            # Complete job
            self.complete_sync_job(
                job_id,
                True,
                result['records_processed'],
                result['records_created'],
                result['records_updated'],
                result['records_failed']
            )

            # Update data source
            self.update_data_source(data_source_id, {
                'status': 'active',
                'last_sync_at': datetime.now().isoformat(),
                'next_sync_at': self._calculate_next_sync(ds_result.data if self.master_client else {}),
                'total_records_synced': (ds_result.data.get('total_records_synced', 0) if self.master_client else 0) + result['records_created'] + result['records_updated'],
                'last_record_count': result['records_processed'],
                'health_status': 'healthy',
                'last_health_check': datetime.now().isoformat(),
                'error_count': 0
            })

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Sync error for {data_source_id}: {error_msg}")

            result['error'] = error_msg

            # Mark job as failed
            if job_id:
                self.complete_sync_job(
                    job_id,
                    False,
                    result['records_processed'],
                    result['records_created'],
                    result['records_updated'],
                    result['records_failed'],
                    error_msg
                )

            # Update data source with error
            self.update_data_source(data_source_id, {
                'status': 'error',
                'error_message': error_msg,
                'error_count': (data_source.get('error_count', 0) or 0) + 1,
                'health_status': 'unhealthy',
                'last_health_check': datetime.now().isoformat()
            })

        return result

    def _calculate_next_sync(self, data_source: Dict) -> str:
        """Calculate next sync timestamp based on frequency."""
        frequency = data_source.get('sync_frequency', 'daily')

        intervals = {
            'realtime': timedelta(minutes=5),
            'hourly': timedelta(hours=1),
            'daily': timedelta(days=1),
            'weekly': timedelta(weeks=1),
            'monthly': timedelta(days=30)
        }

        interval = intervals.get(frequency, timedelta(days=1))
        return (datetime.now() + interval).isoformat()

    def process_webhooks(self) -> Dict:
        """Process pending webhook payloads."""
        payloads = self.webhook_receiver.get_pending()

        result = {
            'processed': 0,
            'success': 0,
            'failed': 0
        }

        for payload in payloads:
            result['processed'] += 1

            try:
                source = payload.get('source')
                data = payload.get('payload', {})

                # Find the data source configuration for this webhook
                if self.master_client:
                    ds_result = self.master_client.table('data_sources').select('*').eq(
                        'source_type', 'webhook'
                    ).execute()

                    for ds in (ds_result.data or []):
                        config = ds.get('connection_config', {})
                        if config.get('webhook_source') == source:
                            # Process this webhook
                            self.sync_data_source({
                                'data_source_id': ds['id'],
                                'client_id': ds['client_id'],
                                'source_type': 'webhook',
                                'connection_config': config
                            })
                            result['success'] += 1
                            break

            except Exception as e:
                logger.error(f"Webhook processing error: {e}")
                result['failed'] += 1

        return result

    def run_sync_cycle(self):
        """Run a single sync cycle for all pending sources."""
        logger.info("Starting sync cycle...")

        # Get pending syncs
        pending = self.get_pending_syncs()
        logger.info(f"Found {len(pending)} data sources due for sync")

        results = {
            'total': len(pending),
            'success': 0,
            'failed': 0,
            'records_synced': 0
        }

        # Process each data source
        futures = []
        for ds in pending:
            future = self.executor.submit(self.sync_data_source, ds)
            futures.append(future)

        # Wait for all to complete
        for future in as_completed(futures):
            try:
                result = future.result()
                if result['success']:
                    results['success'] += 1
                    results['records_synced'] += result['records_created'] + result['records_updated']
                else:
                    results['failed'] += 1
            except Exception as e:
                logger.error(f"Sync future error: {e}")
                results['failed'] += 1

        # Process webhooks
        webhook_results = self.process_webhooks()
        results['webhooks_processed'] = webhook_results['processed']

        logger.info(f"Sync cycle complete: {results}")
        return results

    def start(self, interval_minutes: int = 5):
        """Start the sync service with scheduled runs."""
        self.running = True

        # Schedule sync cycles
        schedule.every(interval_minutes).minutes.do(self.run_sync_cycle)

        # Initial run
        self.run_sync_cycle()

        # Keep running
        logger.info(f"Sync service started (interval: {interval_minutes} minutes)")
        while self.running:
            schedule.run_pending()
            time.sleep(1)

    def stop(self):
        """Stop the sync service."""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Sync service stopped")


# ============================================================================
# FLASK ROUTES FOR SYNC MANAGEMENT
# ============================================================================

def register_sync_routes(app, sync_service: DataSyncService):
    """Register Flask routes for sync management."""

    @app.route('/api/sync/trigger', methods=['POST'])
    def trigger_sync():
        """Manually trigger a sync for a specific data source."""
        data = request.get_json()
        if not data or 'data_source_id' not in data:
            return jsonify({'success': False, 'error': 'data_source_id required'}), 400

        data_source_id = data['data_source_id']

        # Get data source info
        if sync_service.master_client:
            ds = sync_service.master_client.table('data_sources').select('*').eq('id', data_source_id).single().execute()

            if not ds.data:
                return jsonify({'success': False, 'error': 'Data source not found'}), 404

            # Trigger sync
            result = sync_service.sync_data_source({
                'data_source_id': data_source_id,
                'client_id': ds.data['client_id'],
                'source_type': ds.data['source_type'],
                'connection_config': ds.data.get('connection_config', {})
            })

            return jsonify({
                'success': result['success'],
                'records_synced': result['records_created'] + result['records_updated'],
                'records_failed': result['records_failed'],
                'error': result.get('error')
            }), 200 if result['success'] else 500

        return jsonify({'success': False, 'error': 'Service not configured'}), 503

    @app.route('/api/sync/status', methods=['GET'])
    def get_sync_status():
        """Get sync status for all data sources or a specific one."""
        data_source_id = request.args.get('data_source_id')

        if sync_service.master_client:
            query = sync_service.master_client.table('data_sources').select(
                'id, name, source_type, status, health_status, last_sync_at, next_sync_at, '
                'error_message, total_records_synced, sync_frequency'
            )

            if data_source_id:
                query = query.eq('id', data_source_id)

            result = query.execute()

            return jsonify({
                'success': True,
                'data_sources': result.data or []
            }), 200

        return jsonify({'success': False, 'error': 'Service not configured'}), 503

    @app.route('/api/sync/jobs', methods=['GET'])
    def get_sync_jobs():
        """Get recent sync jobs."""
        data_source_id = request.args.get('data_source_id')
        limit = request.args.get('limit', 20, type=int)

        if sync_service.master_client:
            query = sync_service.master_client.table('sync_jobs').select(
                'id, data_source_id, job_type, status, started_at, completed_at, '
                'duration_ms, records_processed, records_created, records_updated, '
                'records_failed, progress_percent, current_step, error_message'
            )

            if data_source_id:
                query = query.eq('data_source_id', data_source_id)

            result = query.order('started_at', desc=True).limit(limit).execute()

            return jsonify({
                'success': True,
                'jobs': result.data or []
            }), 200

        return jsonify({'success': False, 'error': 'Service not configured'}), 503

    @app.route('/api/webhooks/receive/<source>', methods=['POST'])
    def receive_webhook(source: str):
        """Receive webhook payload from external source."""
        signature = request.headers.get('X-Hub-Signature-256') or request.headers.get('X-Signature')

        try:
            payload = request.get_json()

            if sync_service.webhook_receiver.receive(payload, source, signature):
                # Trigger immediate processing
                sync_service.process_webhooks()
                return jsonify({'success': True, 'message': 'Webhook received'}), 200
            else:
                return jsonify({'success': False, 'error': 'Invalid signature'}), 401

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/sync/health', methods=['GET'])
    def sync_health():
        """Health check for sync service."""
        return jsonify({
            'success': True,
            'status': 'running' if sync_service.running else 'stopped',
            'webhook_queue_size': sync_service.webhook_receiver.payload_queue.qsize(),
            'webhooks_processed': sync_service.webhook_receiver.processed_count
        }), 200


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Test the sync service
    service = DataSyncService()

    print("Testing sync service...")
    print(f"Master DB configured: {service.master_client is not None}")
    print(f"Encryption configured: {service.fernet is not None}")

    # Get pending syncs
    pending = service.get_pending_syncs()
    print(f"Pending syncs: {len(pending)}")

    # Run single cycle
    if pending:
        results = service.run_sync_cycle()
        print(f"Sync results: {results}")

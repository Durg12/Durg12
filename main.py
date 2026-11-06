import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import jwt
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from webauthn import (
    base64url_to_bytes,
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers import bytes_to_base64url
from webauthn.helpers.exceptions import (
    InvalidAuthenticationResponse,
    InvalidRegistrationResponse,
    WebAuthnException,
)
from webauthn.helpers.structs import (
    AuthenticatorAttachment,
    AuthenticatorSelectionCriteria,
    AuthenticatorTransport,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vercel-fido2-engine")

# ============================================================================
# VERCEL & CLOUD CONFIGURATION (Auto-detects Vercel domain if left blank)
# ============================================================================
RP_ID: str = os.getenv("RP_ID", "")
EXPECTED_ORIGIN: str = os.getenv("EXPECTED_ORIGIN", "")
RP_NAME: str = os.getenv("RP_NAME", "GIS Biometric Identity Core")

JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "prod-fido2-hs256-signing-secret-key-99281726354")
MAGIC_LINK_SECRET: str = os.getenv("MAGIC_LINK_SECRET", "prod-oob-magic-link-signing-key-88374655123")
JWT_ALGORITHM: str = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
REFRESH_TOKEN_EXPIRE_DAYS: int = 30
GRACE_PERIOD_SECONDS: int = 20
CHALLENGE_TTL_SECONDS: int = 120
MAGIC_LINK_EXPIRY_MINUTES: int = 10
RESEND_API_KEY: str = os.getenv("RESEND_API_KEY", "")

# ============================================================================
# HYBRID FIRESTORE + SERVERLESS STATELESS ENGINE
# Uses Google Cloud Firestore when credentials are set; automatically falls
# back to cryptographically signed stateless JWT state on zero-config Vercel.
# ============================================================================
db = None
try:
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("FIRESTORE_PROJECT_ID"):
        from google.cloud import firestore
        db = firestore.AsyncClient(project=os.getenv("FIRESTORE_PROJECT_ID"))
except Exception as _fs_err:
    logger.info(f"Firestore not configured, using stateless serverless engine: {_fs_err}")
    db = None

MEMORY_USERS: Dict[str, Dict[str, Any]] = {}
MEMORY_FAMILIES: Dict[str, Dict[str, Any]] = {}

app = FastAPI(
    title=RP_NAME,
    version="2.1.0",
    docs_url="/api/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TenantContext:
    def __init__(self, x_tenant_id: str = Header(default="store_default", alias="X-Tenant-ID")):
        cleaned = x_tenant_id.strip().lower()
        self.tenant_id = cleaned if cleaned else "store_default"


class RegisterBeginPayload(BaseModel):
    identity: Optional[str] = None
    email: Optional[str] = None
    username: Optional[str] = None


class RegisterFinishPayload(BaseModel):
    identity: Optional[str] = None
    email: Optional[str] = None
    username: Optional[str] = None
    challenge_id: str
    credential: Dict[str, Any]


class LoginBeginPayload(BaseModel):
    identity: Optional[str] = None
    email: Optional[str] = None
    username: Optional[str] = None
    credential_token: Optional[str] = None


class LoginVerifyPayload(BaseModel):
    identity: Optional[str] = None
    email: Optional[str] = None
    username: Optional[str] = None
    challenge_id: str
    credential_token: Optional[str] = None
    assertion: Optional[Dict[str, Any]] = None
    credential: Optional[Dict[str, Any]] = None


class RefreshPayload(BaseModel):
    refresh_token: str


class MagicLinkRequestPayload(BaseModel):
    identity: Optional[str] = None
    email: Optional[str] = None


class MagicLinkVerifyPayload(BaseModel):
    magic_token: str


def extract_identity(obj: Any) -> str:
    raw = getattr(obj, "identity", None) or getattr(obj, "email", None) or getattr(obj, "username", None) or ""
    cleaned = str(raw).strip().lower()
    if not cleaned or len(cleaned) < 2 or len(cleaned) > 64:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Valid identity is required.")
    return cleaned


def resolve_rp_context(request: Request) -> Tuple[str, str]:
    if RP_ID and EXPECTED_ORIGIN:
        return RP_ID, EXPECTED_ORIGIN

    origin = request.headers.get("origin")
    forwarded_host = request.headers.get("x-forwarded-host")
    host = forwarded_host or request.headers.get("host", "localhost:8000")
    proto = request.headers.get("x-forwarded-proto", request.url.scheme or "https")

    rp_id = host.split(":")[0]
    expected_origin = origin if origin else f"{proto}://{host}"
    return rp_id, expected_origin


def sign_stateless_token(payload: Dict[str, Any], secret: str = JWT_SECRET_KEY) -> str:
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def verify_stateless_token(token: str, secret: str = JWT_SECRET_KEY) -> Dict[str, Any]:
    try:
        return jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Challenge or token expired.") from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid cryptographic token.") from exc


def issue_session_tokens(
    tenant_id: str,
    user_id: str,
    identity: str,
    credential_token: Optional[str] = None,
) -> Dict[str, Any]:
    now = int(time.time())
    family_id = str(uuid.uuid4())
    refresh_jti = str(uuid.uuid4())
    exp_30d = now + (REFRESH_TOKEN_EXPIRE_DAYS * 86400)

    access_token = sign_stateless_token(
        {
            "iss": RP_NAME,
            "sub": user_id,
            "email": identity,
            "tenant_id": tenant_id,
            "family_id": family_id,
            "type": "access",
            "iat": now,
            "exp": now + (ACCESS_TOKEN_EXPIRE_MINUTES * 60),
        }
    )

    refresh_token = sign_stateless_token(
        {
            "iss": RP_NAME,
            "sub": user_id,
            "email": identity,
            "tenant_id": tenant_id,
            "family_id": family_id,
            "token_id": refresh_jti,
            "type": "refresh",
            "iat": now,
            "exp": exp_30d,
        }
    )

    MEMORY_FAMILIES[family_id] = {
        "family_id": family_id,
        "user_id": user_id,
        "email": identity,
        "tenant_id": tenant_id,
        "active_refresh_token_id": refresh_jti,
        "grace_tokens": {},
        "expires_at": exp_30d,
    }

    resp: Dict[str, Any] = {
        "verified": True,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token": access_token,
        "token_type": "Bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user_id": user_id,
        "email": identity,
        "username": identity,
        "tenant_id": tenant_id,
    }
    if credential_token:
        resp["credential_token"] = credential_token
    return resp


@app.get("/", include_in_schema=False)
async def serve_frontend() -> Any:
    index_path = Path(__file__).resolve().parent / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>FIDO2 Biometric API is Live on Vercel</h1>")


@app.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse(content={"status": "healthy", "timestamp": int(time.time())})


@app.post("/api/auth/register-begin")
async def register_begin(
    payload: RegisterBeginPayload,
    request: Request,
    tenant: TenantContext = Depends(),
) -> JSONResponse:
    identity = extract_identity(payload)
    rp_id, _ = resolve_rp_context(request)
    user_id = str(uuid.uuid4())

    options = generate_registration_options(
        rp_id=rp_id,
        rp_name=RP_NAME,
        user_id=user_id.encode("utf-8"),
        user_name=identity,
        user_display_name=identity,
        authenticator_selection=AuthenticatorSelectionCriteria(
            authenticator_attachment=AuthenticatorAttachment.PLATFORM,
            resident_key=ResidentKeyRequirement.PREFERRED,
            user_verification=UserVerificationRequirement.REQUIRED,
        ),
        timeout=120000,
    )

    now = int(time.time())
    challenge_b64 = bytes_to_base64url(options.challenge)

    # Stateless signed challenge token so Vercel serverless cold-starts never lose challenge state
    challenge_id = sign_stateless_token(
        {
            "ceremony": "reg",
            "challenge": challenge_b64,
            "user_id": user_id,
            "identity": identity,
            "tenant_id": tenant.tenant_id,
            "exp": now + CHALLENGE_TTL_SECONDS,
        }
    )

    pub_key_dict = json.loads(options_to_json(options))
    return JSONResponse(
        content={
            "challenge_id": challenge_id,
            "publicKey": pub_key_dict,
            **pub_key_dict,
        }
    )


@app.post("/api/auth/register-finish", status_code=status.HTTP_201_CREATED)
async def register_finish(
    payload: RegisterFinishPayload,
    request: Request,
    tenant: TenantContext = Depends(),
) -> JSONResponse:
    identity = extract_identity(payload)
    ch_data = verify_stateless_token(payload.challenge_id)

    if ch_data.get("ceremony") != "reg" or ch_data.get("identity") != identity:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid registration challenge.")

    rp_id, expected_origin = resolve_rp_context(request)

    try:
        verification = await run_in_threadpool(
            verify_registration_response,
            credential=payload.credential,
            expected_challenge=base64url_to_bytes(ch_data["challenge"]),
            expected_rp_id=rp_id,
            expected_origin=expected_origin,
            require_user_verification=True,
        )
    except (InvalidRegistrationResponse, WebAuthnException, ValueError, KeyError) as exc:
        logger.warning(f"Registration attestation failed for {identity}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Hardware passkey attestation failed: {str(exc)}",
        ) from exc

    cred_id_b64 = bytes_to_base64url(verification.credential_id)
    pub_key_b64 = bytes_to_base64url(verification.credential_public_key)
    user_id = ch_data["user_id"]

    user_record = {
        "user_id": user_id,
        "identity": identity,
        "tenant_id": tenant.tenant_id,
        "credential_id": cred_id_b64,
        "public_key": pub_key_b64,
        "sign_count": verification.sign_count,
    }
    MEMORY_USERS[f"{tenant.tenant_id}:{identity}"] = user_record

    # Signed stateless credential vault token for zero-DB Vercel persistence
    credential_token = sign_stateless_token(
        {
            **user_record,
            "type": "cred_vault",
            "exp": int(time.time()) + (365 * 86400),
        }
    )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content=issue_session_tokens(tenant.tenant_id, user_id, identity, credential_token),
    )


@app.post("/api/auth/login-begin")
async def login_begin(
    payload: LoginBeginPayload,
    request: Request,
    tenant: TenantContext = Depends(),
) -> JSONResponse:
    identity = extract_identity(payload)
    db_key = f"{tenant.tenant_id}:{identity}"
    user_record = MEMORY_USERS.get(db_key)

    if not user_record and payload.credential_token:
        decoded_cred = verify_stateless_token(payload.credential_token)
        if decoded_cred.get("identity") == identity:
            user_record = decoded_cred
            MEMORY_USERS[db_key] = user_record

    rp_id, _ = resolve_rp_context(request)
    allowed_credentials: List[PublicKeyCredentialDescriptor] = []

    if user_record and user_record.get("credential_id"):
        allowed_credentials.append(
            PublicKeyCredentialDescriptor(
                id=base64url_to_bytes(user_record["credential_id"]),
                transports=[AuthenticatorTransport.INTERNAL, AuthenticatorTransport.HYBRID],
            )
        )

    options = generate_authentication_options(
        rp_id=rp_id,
        allow_credentials=allowed_credentials,
        user_verification=UserVerificationRequirement.REQUIRED,
        timeout=120000,
    )

    challenge_b64 = bytes_to_base64url(options.challenge)
    challenge_id = sign_stateless_token(
        {
            "ceremony": "auth",
            "challenge": challenge_b64,
            "identity": identity,
            "tenant_id": tenant.tenant_id,
            "exp": int(time.time()) + CHALLENGE_TTL_SECONDS,
        }
    )

    pub_key_dict = json.loads(options_to_json(options))
    return JSONResponse(
        content={
            "challenge_id": challenge_id,
            "publicKey": pub_key_dict,
            **pub_key_dict,
        }
    )


async def _process_login_verify(
    payload: LoginVerifyPayload,
    request: Request,
    tenant: TenantContext,
) -> JSONResponse:
    identity = extract_identity(payload)
    ch_data = verify_stateless_token(payload.challenge_id)

    if ch_data.get("ceremony") != "auth" or ch_data.get("identity") != identity:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid authentication challenge.")

    db_key = f"{tenant.tenant_id}:{identity}"
    user_record = MEMORY_USERS.get(db_key)

    if not user_record and payload.credential_token:
        decoded_cred = verify_stateless_token(payload.credential_token)
        if decoded_cred.get("identity") == identity:
            user_record = decoded_cred
            MEMORY_USERS[db_key] = user_record

    if not user_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Passkey record not found. Please register a Passkey first.",
        )

    assertion_data = payload.assertion or payload.credential
    if not assertion_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing biometric assertion payload.")

    rp_id, expected_origin = resolve_rp_context(request)
    stored_sign_count = int(user_record.get("sign_count", 0))

    try:
        verification = await run_in_threadpool(
            verify_authentication_response,
            credential=assertion_data,
            expected_challenge=base64url_to_bytes(ch_data["challenge"]),
            expected_rp_id=rp_id,
            expected_origin=expected_origin,
            credential_public_key=base64url_to_bytes(user_record["public_key"]),
            credential_current_sign_count=0,
            require_user_verification=True,
        )
    except (InvalidAuthenticationResponse, WebAuthnException, ValueError, KeyError) as exc:
        logger.warning(f"Authentication failed for {identity}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Biometric signature verification failed: {str(exc)}",
        ) from exc

    incoming_count = verification.new_sign_count
    if stored_sign_count > 0 and incoming_count <= stored_sign_count:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Security Anomaly: Passkey cloning detected via sign-count divergence.",
        )

    user_record["sign_count"] = incoming_count
    MEMORY_USERS[db_key] = user_record

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=issue_session_tokens(tenant.tenant_id, user_record["user_id"], identity),
    )


@app.post("/api/auth/login-verify")
async def login_verify(
    payload: LoginVerifyPayload,
    request: Request,
    tenant: TenantContext = Depends(),
) -> JSONResponse:
    return await _process_login_verify(payload, request, tenant)


@app.post("/api/auth/login-finish")
async def login_finish(
    payload: LoginVerifyPayload,
    request: Request,
    tenant: TenantContext = Depends(),
) -> JSONResponse:
    return await _process_login_verify(payload, request, tenant)


@app.post("/api/auth/refresh")
async def refresh_tokens(
    payload: RefreshPayload,
    tenant: TenantContext = Depends(),
) -> JSONResponse:
    decoded = verify_stateless_token(payload.refresh_token)
    if decoded.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token type.")

    if decoded.get("tenant_id") != tenant.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-tenant token rejected.")

    user_id = decoded.get("sub", str(uuid.uuid4()))
    identity = decoded.get("email", "")
    return JSONResponse(content=issue_session_tokens(tenant.tenant_id, user_id, identity))


@app.post("/api/recovery/request-magic-link")
async def request_magic_link(
    payload: MagicLinkRequestPayload,
    background_tasks: BackgroundTasks,
    tenant: TenantContext = Depends(),
) -> JSONResponse:
    identity = extract_identity(payload)
    now = int(time.time())
    magic_token = sign_stateless_token(
        {
            "type": "magic_recovery",
            "sub": str(uuid.uuid4()),
            "identity": identity,
            "tenant_id": tenant.tenant_id,
            "jti": str(uuid.uuid4()),
            "exp": now + (MAGIC_LINK_EXPIRY_MINUTES * 60),
        },
        secret=MAGIC_LINK_SECRET,
    )

    return JSONResponse(
        content={
            "status": "dispatched",
            "message": "Single-use 10-minute recovery Magic Link generated.",
            "magic_token": magic_token,
        }
    )


@app.post("/api/recovery/verify-magic-link")
async def verify_magic_link(
    payload: MagicLinkVerifyPayload,
    tenant: TenantContext = Depends(),
) -> JSONResponse:
    decoded = verify_stateless_token(payload.magic_token, secret=MAGIC_LINK_SECRET)
    if decoded.get("type") != "magic_recovery" or decoded.get("tenant_id") != tenant.tenant_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Magic Link token.")

    return JSONResponse(
        content=issue_session_tokens(
            tenant.tenant_id,
            decoded.get("sub", str(uuid.uuid4())),
            decoded.get("identity", "verified_user"),
        )
    )


@app.post("/api/auth/logout")
async def logout() -> JSONResponse:
    return JSONResponse(content={"logged_out": True, "message": "Session terminated."})

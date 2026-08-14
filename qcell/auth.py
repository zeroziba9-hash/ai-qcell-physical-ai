"""Session authentication and role-based permissions for the portfolio console."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import os
from typing import Mapping, MutableMapping


VIEW_OPERATIONS = "view_operations"
RUN_INSPECTION = "run_inspection"
REVIEW_CASES = "review_cases"
MANAGE_CAPA = "manage_capa"
MANAGE_MODELS = "manage_models"

ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "viewer": frozenset({VIEW_OPERATIONS}),
    "operator": frozenset({VIEW_OPERATIONS, RUN_INSPECTION}),
    "quality_manager": frozenset(
        {VIEW_OPERATIONS, RUN_INSPECTION, REVIEW_CASES, MANAGE_CAPA}
    ),
    "admin": frozenset(
        {
            VIEW_OPERATIONS,
            RUN_INSPECTION,
            REVIEW_CASES,
            MANAGE_CAPA,
            MANAGE_MODELS,
        }
    ),
}

DEMO_PASSWORDS = {
    "operator": "qcell-operator",
    "quality": "qcell-quality",
    "admin": "qcell-admin",
}


@dataclass(frozen=True)
class Principal:
    username: str
    display_name: str
    role: str
    authenticated: bool = True

    @property
    def permissions(self) -> frozenset[str]:
        return ROLE_PERMISSIONS.get(self.role, frozenset())

    def can(self, permission: str) -> bool:
        return permission in self.permissions

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Principal":
        return cls(
            username=str(payload.get("username", "guest")),
            display_name=str(payload.get("display_name", "Portfolio Guest")),
            role=str(payload.get("role", "viewer")),
            authenticated=bool(payload.get("authenticated", False)),
        )


GUEST = Principal(
    username="guest",
    display_name="Portfolio Guest",
    role="viewer",
    authenticated=False,
)


@dataclass(frozen=True)
class UserCredential:
    username: str
    display_name: str
    role: str
    salt: str
    password_hash: str


def hash_password(password: str, salt: str, *, iterations: int = 180_000) -> str:
    if not password:
        raise ValueError("password must not be empty")
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()


def _demo_credentials() -> list[UserCredential]:
    definitions = (
        ("operator", "Line Operator", "operator", DEMO_PASSWORDS["operator"]),
        ("quality", "Quality Manager", "quality_manager", DEMO_PASSWORDS["quality"]),
        ("admin", "System Administrator", "admin", DEMO_PASSWORDS["admin"]),
    )
    return [
        UserCredential(
            username=username,
            display_name=display_name,
            role=role,
            salt=f"ai-qcell-demo-{username}-2026",
            password_hash=hash_password(password, f"ai-qcell-demo-{username}-2026"),
        )
        for username, display_name, role, password in definitions
    ]


class CredentialStore:
    def __init__(self, credentials: list[UserCredential], *, demo_mode: bool) -> None:
        self._credentials = {credential.username: credential for credential in credentials}
        self.demo_mode = demo_mode

    @classmethod
    def from_environment(cls) -> "CredentialStore":
        mode = os.getenv("QCELL_AUTH_MODE", "demo").strip().lower()
        raw_users = os.getenv("QCELL_USERS_JSON", "").strip()
        if raw_users:
            payload = json.loads(raw_users)
            credentials = [
                UserCredential(
                    username=str(item["username"]),
                    display_name=str(item.get("display_name", item["username"])),
                    role=str(item["role"]),
                    salt=str(item["salt"]),
                    password_hash=str(item["password_hash"]),
                )
                for item in payload
            ]
        elif mode == "demo":
            credentials = _demo_credentials()
        else:
            credentials = []
        for credential in credentials:
            if credential.role not in ROLE_PERMISSIONS:
                raise ValueError(f"unsupported role: {credential.role}")
        return cls(credentials, demo_mode=mode == "demo")

    def authenticate(self, username: str, password: str) -> Principal | None:
        credential = self._credentials.get(username.strip())
        if credential is None or not password:
            return None
        candidate = hash_password(password, credential.salt)
        if not hmac.compare_digest(candidate, credential.password_hash):
            return None
        return Principal(
            username=credential.username,
            display_name=credential.display_name,
            role=credential.role,
        )

    def usernames(self) -> tuple[str, ...]:
        return tuple(sorted(self._credentials))


SESSION_KEY = "qcell_principal"


def session_principal(state: Mapping[str, object]) -> Principal:
    payload = state.get(SESSION_KEY)
    if isinstance(payload, Mapping):
        principal = Principal.from_dict(payload)
        if principal.role in ROLE_PERMISSIONS:
            return principal
    return GUEST


def login_session(state: MutableMapping[str, object], principal: Principal) -> None:
    if not principal.authenticated:
        raise ValueError("cannot log in an unauthenticated principal")
    state[SESSION_KEY] = principal.to_dict()


def logout_session(state: MutableMapping[str, object]) -> None:
    state.pop(SESSION_KEY, None)


def has_permission(state: Mapping[str, object], permission: str) -> bool:
    return session_principal(state).can(permission)

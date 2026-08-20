import json

from qcell.auth import (
    GUEST,
    MANAGE_CAPA,
    MANAGE_MODELS,
    REVIEW_CASES,
    RUN_INSPECTION,
    CredentialStore,
    Principal,
    hash_password,
    has_permission,
    login_session,
    logout_session,
    session_principal,
)


def test_demo_accounts_authenticate_and_enforce_role_permissions(monkeypatch) -> None:
    monkeypatch.setenv("QCELL_AUTH_MODE", "demo")
    monkeypatch.delenv("QCELL_USERS_JSON", raising=False)
    store = CredentialStore.from_environment()

    operator = store.authenticate("operator", "qcell-operator")
    quality = store.authenticate("quality", "qcell-quality")
    admin = store.authenticate("admin", "qcell-admin")

    assert operator is not None and operator.can(RUN_INSPECTION)
    assert operator.can(REVIEW_CASES) is False
    assert quality is not None and quality.can(REVIEW_CASES)
    assert quality.can(MANAGE_CAPA)
    assert quality.can(MANAGE_MODELS) is False
    assert admin is not None and admin.can(MANAGE_MODELS)
    assert store.authenticate("admin", "wrong") is None


def test_session_login_logout_round_trip() -> None:
    state: dict[str, object] = {}
    principal = Principal("quality", "Quality Manager", "quality_manager")

    assert session_principal(state) == GUEST
    login_session(state, principal)
    assert session_principal(state) == principal
    assert has_permission(state, MANAGE_CAPA)
    logout_session(state)
    assert session_principal(state) == GUEST


def test_strict_environment_credentials_are_loaded(monkeypatch) -> None:
    salt = "strict-user-salt"
    payload = [
        {
            "username": "factory-admin",
            "display_name": "Factory Admin",
            "role": "admin",
            "salt": salt,
            "password_hash": hash_password("secret-pass", salt),
        }
    ]
    monkeypatch.setenv("QCELL_AUTH_MODE", "strict")
    monkeypatch.setenv("QCELL_USERS_JSON", json.dumps(payload))

    store = CredentialStore.from_environment()

    assert store.demo_mode is False
    assert store.authenticate("factory-admin", "secret-pass") is not None
    assert store.authenticate("admin", "qcell-admin") is None

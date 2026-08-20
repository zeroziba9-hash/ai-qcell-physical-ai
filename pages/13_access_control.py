from __future__ import annotations

import json
import secrets

import pandas as pd
import streamlit as st

from qcell.auth import (
    ROLE_PERMISSIONS,
    CredentialStore,
    hash_password,
    login_session,
    logout_session,
    session_principal,
)
from qcell.ui import inject_global_css, page_header, section_header, status_strip, workflow_strip


st.set_page_config(page_title="접근 제어 · AI-QCell", page_icon="🔐", layout="wide")
inject_global_css()

credentials = CredentialStore.from_environment()
principal = session_principal(st.session_state)

page_header(
    "SECURITY · ROLE BASED ACCESS CONTROL",
    "접근 제어 센터",
    "검사자, 품질 관리자와 시스템 관리자의 세션 권한을 분리하고 주요 운영 변경을 통제합니다.",
    status="RBAC POLICY ACTIVE",
)
status_strip(
    [
        {
            "label": "Session",
            "value": "AUTHENTICATED" if principal.authenticated else "GUEST",
            "tone": "good" if principal.authenticated else "warn",
        },
        {"label": "Identity", "value": principal.display_name, "tone": "good"},
        {"label": "Role", "value": principal.role.upper(), "tone": "good"},
        {
            "label": "Auth mode",
            "value": "DEMO" if credentials.demo_mode else "STRICT",
            "tone": "warn" if credentials.demo_mode else "good",
        },
    ]
)
workflow_strip(["자격 증명 확인", "세션 발급", "권한 평가", "운영 권한 적용"])

login_column, policy_column = st.columns([0.43, 0.57], gap="large")
with login_column:
    section_header(
        "세션 로그인",
        "로그인 후 역할에 허용된 변경 작업만 활성화됩니다.",
        code="IDENTITY / 01",
    )
    if principal.authenticated:
        st.success(f"{principal.display_name} 계정으로 로그인했습니다.")
        st.write(f"사용자: `{principal.username}`")
        st.write(f"역할: `{principal.role}`")
        if st.button("로그아웃", type="primary", width="stretch"):
            logout_session(st.session_state)
            st.rerun()
    else:
        with st.form("qcell-login", clear_on_submit=False):
            username = st.text_input("사용자 이름")
            password = st.text_input("비밀번호", type="password")
            submitted = st.form_submit_button("로그인", type="primary", width="stretch")
        if submitted:
            authenticated = credentials.authenticate(username, password)
            if authenticated is None:
                st.error("사용자 이름 또는 비밀번호가 올바르지 않습니다.")
            else:
                login_session(st.session_state, authenticated)
                st.rerun()
        if credentials.demo_mode:
            st.info(
                "데모 계정 · operator / qcell-operator · quality / qcell-quality · "
                "admin / qcell-admin"
            )
        elif not credentials.usernames():
            st.warning("QCELL_USERS_JSON에 운영 계정을 등록해야 로그인이 가능합니다.")

with policy_column:
    section_header(
        "역할 정책",
        "읽기 전용 포트폴리오 접근과 생산 변경 권한을 분리합니다.",
        code="POLICY MATRIX / 02",
    )
    permissions = sorted({permission for values in ROLE_PERMISSIONS.values() for permission in values})
    matrix = []
    for role, allowed in ROLE_PERMISSIONS.items():
        matrix.append(
            {
                "역할": role,
                **{permission: "ALLOW" if permission in allowed else "—" for permission in permissions},
            }
        )
    st.dataframe(pd.DataFrame(matrix), hide_index=True, width="stretch")

section_header(
    "운영 계정 구성",
    "배포 환경에서는 평문 비밀번호 대신 PBKDF2 해시가 포함된 QCELL_USERS_JSON을 사용합니다.",
    code="STRICT MODE / 03",
)
st.code(
    """QCELL_AUTH_MODE="strict"\nQCELL_USERS_JSON='[{"username":"...","role":"admin","salt":"...","password_hash":"..."}]'""",
    language="bash",
)

if principal.role == "admin":
    with st.expander("운영 계정 해시 생성기"):
        account_a, account_b, account_c = st.columns(3)
        with account_a:
            new_username = st.text_input("새 사용자", key="rbac-new-user")
        with account_b:
            new_role = st.selectbox("새 역할", list(ROLE_PERMISSIONS), key="rbac-new-role")
        with account_c:
            new_display_name = st.text_input("표시 이름", key="rbac-display-name")
        new_password = st.text_input("초기 비밀번호", type="password", key="rbac-new-password")
        if st.button("해시 구성 생성", width="stretch"):
            if not new_username or not new_password:
                st.warning("사용자 이름과 초기 비밀번호를 입력하세요.")
            else:
                salt = secrets.token_hex(16)
                st.session_state.rbac_generated_user = {
                    "username": new_username,
                    "display_name": new_display_name or new_username,
                    "role": new_role,
                    "salt": salt,
                    "password_hash": hash_password(new_password, salt),
                }
        generated = st.session_state.get("rbac_generated_user")
        if generated:
            st.code(json.dumps([generated], ensure_ascii=False, indent=2), language="json")

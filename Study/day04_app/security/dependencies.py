"""FastAPI API Key 认证与 RBAC 依赖。"""

import hashlib
import hmac
import json
from collections.abc import Callable
from typing import Any

from fastapi import Depends, Header, Request, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from day04_app.common.exceptions import AuthenticationException, AuthorizationException
from day04_app.database import get_db
from day04_app.security.permissions import (
    KNOWN_AI_PERMISSIONS,
    KNOWN_AI_ROLES,
    permissions_for_roles,
)
from day04_app.security.principal import DEVELOPMENT_PRINCIPAL, SecurityPrincipal
from day04_app.services.security_audit_service import create_security_audit
from settings import settings


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
service_api_key_header = APIKeyHeader(name="X-Service-API-Key", auto_error=False)


def hash_api_key(raw_api_key: str) -> str:
    return hashlib.sha256(raw_api_key.encode("utf-8")).hexdigest()


def _load_key_records(raw_value: str, setting_name: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{setting_name} 不是合法 JSON") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{setting_name} 必须是 JSON 数组")
    return payload


def _load_api_key_records() -> list[dict[str, Any]]:
    """本地直连调用者 Key；只用于教学或没有 Java 网关的受控场景。"""
    return _load_key_records(settings.security_api_keys_json, "SECURITY_API_KEYS_JSON")


def _load_service_key_records() -> list[dict[str, Any]]:
    """Java 等受信任服务的凭据，不承载最终用户身份。"""
    return _load_key_records(
        settings.security_service_api_keys_json,
        "SECURITY_SERVICE_API_KEYS_JSON",
    )


def _validate_key_hash(record: dict[str, Any], setting_name: str) -> None:
    key_hash = str(record.get("key_hash", ""))
    if len(key_hash) != 64 or any(
        char not in "0123456789abcdefABCDEF" for char in key_hash
    ):
        raise ValueError(f"{setting_name} 配置中的 key_hash 必须是 SHA-256 十六进制值")


def validate_security_configuration() -> None:
    if not settings.security_enabled:
        return
    api_key_records = _load_api_key_records()
    service_key_records = _load_service_key_records()
    if not api_key_records and not service_key_records:
        raise ValueError(
            "SECURITY_ENABLED=true 时至少配置 SECURITY_API_KEYS_JSON 或 "
            "SECURITY_SERVICE_API_KEYS_JSON"
        )
    for record in api_key_records:
        required = ("key_id", "key_hash", "actor_id", "roles")
        if any(not record.get(field) for field in required):
            raise ValueError("API Key 配置缺少 key_id/key_hash/actor_id/roles")
        _validate_key_hash(record, "SECURITY_API_KEYS_JSON")
    for record in service_key_records:
        required = ("key_id", "key_hash", "service_id")
        if any(not record.get(field) for field in required):
            raise ValueError(
                "服务间 Key 配置缺少 key_id/key_hash/service_id"
            )
        _validate_key_hash(record, "SECURITY_SERVICE_API_KEYS_JSON")


def _authenticate(raw_api_key: str) -> SecurityPrincipal | None:
    incoming_hash = hash_api_key(raw_api_key)
    for record in _load_api_key_records():
        expected_hash = str(record.get("key_hash", "")).lower()
        if not hmac.compare_digest(incoming_hash, expected_hash):
            continue
        roles = tuple(str(item) for item in record.get("roles", []))
        explicit_permissions = frozenset(
            str(item) for item in record.get("permissions", [])
        )
        return SecurityPrincipal(
            actor_id=str(record["actor_id"]),
            api_key_id=str(record["key_id"]),
            roles=roles,
            permissions=permissions_for_roles(roles) | explicit_permissions,
        )
    return None


def _authenticate_service(raw_api_key: str) -> dict[str, Any] | None:
    """只确认调用方是受信任服务，不把服务 Key 伪装成某个最终用户。"""
    incoming_hash = hash_api_key(raw_api_key)
    for record in _load_service_key_records():
        expected_hash = str(record.get("key_hash", "")).lower()
        if hmac.compare_digest(incoming_hash, expected_hash):
            return record
    return None


def _parse_csv_header(raw_value: str | None, header_name: str) -> tuple[str, ...]:
    if not raw_value:
        return ()
    values = tuple(item.strip() for item in raw_value.split(",") if item.strip())
    if not values:
        raise ValueError(f"{header_name} 不能为空")
    return values


def _parse_data_scope(raw_value: str | None) -> dict[str, Any]:
    """数据范围只作为业务过滤条件透传，必须是小型 JSON 对象。"""
    if not raw_value:
        return {}
    if len(raw_value) > 4096:
        raise ValueError("X-AI-Data-Scope 不能超过 4096 个字符")
    try:
        data_scope = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError("X-AI-Data-Scope 必须是合法 JSON") from exc
    if not isinstance(data_scope, dict) or any(
        not isinstance(key, str) or not key.strip() for key in data_scope
    ):
        raise ValueError("X-AI-Data-Scope 必须是键名非空的 JSON 对象")
    return data_scope


def _build_trusted_proxy_principal(
    *,
    service_record: dict[str, Any],
    actor_id: str | None,
    raw_roles: str | None,
    raw_permissions: str | None,
    raw_data_scope: str | None,
) -> SecurityPrincipal:
    """由已认证 Java 服务传入的用户上下文构造 Principal。

    这里不会解析浏览器 token。Java 需要先完成自身登录和 URI/Method 权限校验，
    再用服务间 Key 证明“这份用户上下文确实来自受信任后端”。
    """
    normalized_actor_id = (actor_id or "").strip()
    if not normalized_actor_id or len(normalized_actor_id) > 64:
        raise ValueError("X-AI-Actor-Id 必须是 1 到 64 位")
    roles = _parse_csv_header(raw_roles, "X-AI-Roles")
    permissions = _parse_csv_header(raw_permissions, "X-AI-Permissions")
    unknown_roles = set(roles) - KNOWN_AI_ROLES
    unknown_permissions = set(permissions) - KNOWN_AI_PERMISSIONS
    if unknown_roles:
        raise ValueError("X-AI-Roles 包含未登记的 AI 角色")
    if unknown_permissions:
        raise ValueError("X-AI-Permissions 包含未登记的 AI 权限")
    if not roles and not permissions:
        raise ValueError("Java 透传身份必须包含 AI 角色或 AI 权限")
    return SecurityPrincipal(
        actor_id=normalized_actor_id,
        api_key_id=str(service_record["key_id"]),
        roles=roles,
        permissions=permissions_for_roles(roles) | frozenset(permissions),
        auth_type="java_trusted_proxy",
        data_scope=_parse_data_scope(raw_data_scope),
        source_service_id=str(service_record["service_id"]),
    )


def _audit_authentication_denied(
    db: Session,
    request: Request,
    reason: str,
    credential_id: str | None = None,
) -> None:
    create_security_audit(
        db,
        trace_id=getattr(request.state, "trace_id", None),
        principal=None,
        permission="authenticated",
        http_method=request.method,
        request_path=request.url.path,
        decision="deny",
        reason=reason,
        credential_id=credential_id,
    )


def get_current_principal(
    request: Request,
    raw_api_key: str | None = Security(api_key_header),
    raw_service_api_key: str | None = Security(service_api_key_header),
    actor_id: str | None = Header(None, alias="X-AI-Actor-Id"),
    raw_roles: str | None = Header(None, alias="X-AI-Roles"),
    raw_permissions: str | None = Header(None, alias="X-AI-Permissions"),
    raw_data_scope: str | None = Header(None, alias="X-AI-Data-Scope"),
    db: Session = Depends(get_db),
) -> SecurityPrincipal:
    if not settings.security_enabled:
        request.state.principal = DEVELOPMENT_PRINCIPAL
        return DEVELOPMENT_PRINCIPAL
    if raw_service_api_key:
        service_record = _authenticate_service(raw_service_api_key)
        if service_record is None:
            _audit_authentication_denied(db, request, "INVALID_SERVICE_API_KEY")
            raise AuthenticationException("X-Service-API-Key 无效")
        try:
            principal = _build_trusted_proxy_principal(
                service_record=service_record,
                actor_id=actor_id,
                raw_roles=raw_roles,
                raw_permissions=raw_permissions,
                raw_data_scope=raw_data_scope,
            )
        except ValueError:
            _audit_authentication_denied(
                db,
                request,
                "INVALID_TRUSTED_PRINCIPAL",
                credential_id=str(service_record["key_id"]),
            )
            raise AuthenticationException("Java 透传的用户上下文不合法") from None
        request.state.principal = principal
        return principal
    if actor_id or raw_roles or raw_permissions or raw_data_scope:
        # 身份头没有服务凭据时一律不可信，避免浏览器伪造 Java 用户上下文。
        _audit_authentication_denied(db, request, "UNTRUSTED_PRINCIPAL_HEADERS")
        raise AuthenticationException("X-AI 用户上下文只能由受信任 Java 服务透传")
    if not raw_api_key:
        _audit_authentication_denied(db, request, "MISSING_API_KEY")
        raise AuthenticationException("缺少 X-API-Key 请求头")

    principal = _authenticate(raw_api_key)
    if principal is None:
        _audit_authentication_denied(db, request, "INVALID_API_KEY")
        raise AuthenticationException("X-API-Key 无效")
    request.state.principal = principal
    return principal


def require_permissions(
    *permissions: str,
    resource_type: str | None = None,
    resource_param: str | None = None,
) -> Callable[..., SecurityPrincipal]:
    if not permissions:
        raise ValueError("require_permissions 至少需要一个权限")
    permission_text = ",".join(permissions)

    def dependency(
        request: Request,
        db: Session = Depends(get_db),
        principal: SecurityPrincipal = Depends(get_current_principal),
    ) -> SecurityPrincipal:
        resource_id = (
            str(request.path_params.get(resource_param))
            if resource_param and request.path_params.get(resource_param) is not None
            else None
        )
        allowed = principal.has_permissions(*permissions)
        if settings.security_enabled:
            create_security_audit(
                db,
                trace_id=getattr(request.state, "trace_id", None),
                principal=principal,
                permission=permission_text,
                http_method=request.method,
                request_path=request.url.path,
                decision="allow" if allowed else "deny",
                reason="PERMISSION_GRANTED" if allowed else "PERMISSION_DENIED",
                resource_type=resource_type,
                resource_id=resource_id,
            )
        if not allowed:
            raise AuthorizationException(f"缺少权限：{permission_text}")
        return principal

    return dependency

"""认证后的调用者上下文，类似 Spring Security Authentication。"""

from dataclasses import dataclass, field
from typing import Any, Mapping

from day04_app.security.permissions import PERMISSION_ALL


@dataclass(frozen=True, slots=True)
class SecurityPrincipal:
    actor_id: str
    api_key_id: str
    roles: tuple[str, ...]
    permissions: frozenset[str]
    auth_type: str = "api_key"
    # 数据范围由已认证的 Java 服务生成，例如园区、企业、部门 ID；不写入授权审计表。
    data_scope: Mapping[str, Any] = field(default_factory=dict)
    # 仅 Java 服务间调用时有值，用于异步任务追溯调用来源。
    source_service_id: str | None = None

    def has_permissions(self, *required_permissions: str) -> bool:
        return PERMISSION_ALL in self.permissions or all(
            permission in self.permissions for permission in required_permissions
        )

    def to_snapshot(self) -> dict[str, Any]:
        """异步任务只传播已认证身份快照，绝不传播原始 API Key。"""
        return {
            "actor_id": self.actor_id,
            "api_key_id": self.api_key_id,
            "roles": list(self.roles),
            "permissions": sorted(self.permissions),
            "auth_type": self.auth_type,
            "data_scope": dict(self.data_scope),
            "source_service_id": self.source_service_id,
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any] | None) -> "SecurityPrincipal":
        if not payload:
            return SYSTEM_PRINCIPAL
        return cls(
            actor_id=str(payload["actor_id"]),
            api_key_id=str(payload.get("api_key_id") or "unknown"),
            roles=tuple(str(item) for item in payload.get("roles", [])),
            permissions=frozenset(str(item) for item in payload.get("permissions", [])),
            auth_type=str(payload.get("auth_type") or "async_snapshot"),
            data_scope=dict(payload.get("data_scope") or {}),
            source_service_id=(
                str(payload["source_service_id"])
                if payload.get("source_service_id")
                else None
            ),
        )


SYSTEM_PRINCIPAL = SecurityPrincipal(
    actor_id="system-worker",
    api_key_id="internal",
    roles=("system",),
    permissions=frozenset({PERMISSION_ALL}),
    auth_type="internal",
)

DEVELOPMENT_PRINCIPAL = SecurityPrincipal(
    actor_id="local-development",
    api_key_id="security-disabled",
    roles=("admin",),
    permissions=frozenset({PERMISSION_ALL}),
    auth_type="development_bypass",
)

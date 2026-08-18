"""Day27 零成本安全冒烟测试：不调用模型、Embedding 或 Reranker。"""

import json
import sys
from pathlib import Path

# 直接执行 scripts/ 下文件时，显式把项目根目录加入模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from day04_app.main import app
from day04_app.security.dependencies import hash_api_key
from day04_app.security.permissions import (
    PERMISSION_TOOL_EXECUTE,
    permissions_for_roles,
)
from day04_app.security.principal import SecurityPrincipal
from day04_app.services.tool_calling_service import (
    ToolDecision,
    execute_registered_tool,
)
from settings import settings


def main() -> None:
    original_enabled = settings.security_enabled
    original_keys = settings.security_api_keys_json
    original_service_keys = settings.security_service_api_keys_json
    settings.security_enabled = True
    settings.security_api_keys_json = json.dumps(
        [
            {
                "key_id": "day27-viewer",
                "key_hash": hash_api_key("viewer-test-key"),
                "actor_id": "learner-viewer",
                "roles": ["viewer"],
            },
            {
                "key_id": "day27-operator",
                "key_hash": hash_api_key("operator-test-key"),
                "actor_id": "learner-operator",
                "roles": ["operator"],
            },
            {
                "key_id": "day27-admin",
                "key_hash": hash_api_key("admin-test-key"),
                "actor_id": "learner-admin",
                "roles": ["admin"],
            },
        ],
        ensure_ascii=False,
    )
    settings.security_service_api_keys_json = json.dumps(
        [
            {
                "key_id": "day27-green-parkplat-service",
                "key_hash": hash_api_key("green-parkplat-service-test-key"),
                "service_id": "green-parkplat",
            }
        ],
        ensure_ascii=False,
    )
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            missing = client.get("/api/chat/prompt-versions")
            denied = client.get(
                "/api/chat/prompt-versions",
                headers={"X-API-Key": "viewer-test-key"},
            )
            allowed = client.get(
                "/api/chat/prompt-versions",
                headers={"X-API-Key": "admin-test-key"},
            )
            audit_denied = client.get(
                "/api/chat/security-audits",
                headers={"X-API-Key": "viewer-test-key"},
            )
            audit_allowed = client.get(
                "/api/chat/security-audits?actor_id=learner-viewer",
                headers={"X-API-Key": "admin-test-key"},
            )
            # 模拟 Java 已完成登录和 URI 权限校验后，使用服务凭据透传 AI Principal。
            gateway_headers = {
                "X-Service-API-Key": "green-parkplat-service-test-key",
                "X-AI-Actor-Id": "park-user-1001",
                "X-AI-Roles": "operator",
                "X-AI-Data-Scope": '{"park_ids":["PARK_001"]}',
            }
            gateway_allowed = client.get(
                "/api/chat/prompt-versions", headers=gateway_headers
            )
            untrusted_context = client.get(
                "/api/chat/prompt-versions",
                headers={
                    "X-API-Key": "viewer-test-key",
                    "X-AI-Actor-Id": "forged-user",
                    "X-AI-Roles": "admin",
                },
            )

            # 同一份经过服务 Key 认证的 Java 上下文中，会话必须按 actor_id 隔离。
            actor_a_headers = {
                "X-Service-API-Key": "green-parkplat-service-test-key",
                "X-AI-Actor-Id": "park-user-session-a",
                "X-AI-Permissions": "ai:invoke",
            }
            actor_b_headers = {
                "X-Service-API-Key": "green-parkplat-service-test-key",
                "X-AI-Actor-Id": "park-user-session-b",
                "X-AI-Permissions": "ai:invoke",
            }
            created_session = client.post("/api/chat/sessions", headers=actor_a_headers)
            session_id = created_session.json()["data"]["session_id"]
            cross_actor_session = client.get(
                f"/api/chat/sessions/{session_id}/messages",
                headers=actor_b_headers,
            )

        assert missing.status_code == 401
        assert missing.json()["code"] == 40101
        assert denied.status_code == 403
        assert denied.json()["code"] == 40301
        assert allowed.status_code == 200
        assert audit_denied.status_code == 403
        assert audit_allowed.status_code == 200
        assert gateway_allowed.status_code == 200
        assert untrusted_context.status_code == 401
        assert created_session.status_code == 200
        assert cross_actor_session.status_code == 403
        audit_items = audit_allowed.json()["data"]["items"]
        assert any(
            item["actor_id"] == "learner-viewer" and item["decision"] == "deny"
            for item in audit_items
        )

        # 即使操作者拥有 tool:execute，高风险写工具仍必须走人工确认，RBAC 不能绕过风险策略。
        operator = SecurityPrincipal(
            actor_id="learner-operator",
            api_key_id="day27-operator",
            roles=("operator",),
            permissions=permissions_for_roles(("operator",)),
        )
        assert operator.has_permissions(PERMISSION_TOOL_EXECUTE)
        decision = ToolDecision(
            need_tool=True,
            tool_name="close_work_order_demo",
            arguments={"business_id": "WO_DAY27", "close_reason": "安全冒烟测试"},
            reason="test",
        )
        tool_result = execute_registered_tool(None, decision, principal=operator)
        assert tool_result is not None
        assert tool_result["status"] == "require_confirm"

        print("DAY27_SECURITY_SMOKE_OK")
        print("missing_api_key=401 insufficient_permission=403 admin_allowed=200")
        print("java_trusted_proxy=200 forged_context=401")
        print("session_ownership_isolated=403 authorization_audit_persisted=true high_risk_tool=require_confirm")
    finally:
        settings.security_enabled = original_enabled
        settings.security_api_keys_json = original_keys
        settings.security_service_api_keys_json = original_service_keys


if __name__ == "__main__":
    main()

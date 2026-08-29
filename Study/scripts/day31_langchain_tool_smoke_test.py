"""Day31：验证 LangChain Tool 只适配工具契约，不能绕过项目策略层。"""

import json
from pathlib import Path
import sys

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day04_app.security.permissions import PERMISSION_TOOL_EXECUTE
from day04_app.security.principal import SecurityPrincipal
from day04_app.services.langchain_tool_adapter_service import (
    build_langchain_tools,
    get_langchain_tool,
)


def _principal(*permissions: str) -> SecurityPrincipal:
    return SecurityPrincipal(
        actor_id="langchain-tool-smoke",
        api_key_id="test-key",
        roles=("operator",),
        permissions=frozenset(permissions),
        auth_type="test",
    )


def main() -> None:
    # 高风险工具会在策略层提前返回 require_confirm，因此不访问真实数据库。
    privileged_catalog = build_langchain_tools(
        object(),  # type: ignore[arg-type]  # 本测试不会进入 executor。
        principal=_principal(PERMISSION_TOOL_EXECUTE),
    )
    close_work_order = get_langchain_tool(privileged_catalog, "close_work_order_demo")
    high_risk_result = json.loads(
        close_work_order.invoke(
            {"business_id": "WO-1001", "close_reason": "现场已处理完成，申请关闭工单"}
        )
    )
    assert high_risk_result["status"] == "require_confirm"
    assert "WRITE_TOOL_REQUIRE_CONFIRM" in high_risk_result["matched_rules"]

    # 就算模型或调用方直接 invoke Tool，缺少权限仍会被项目策略层 block。
    no_permission_catalog = build_langchain_tools(
        object(),  # type: ignore[arg-type]
        principal=_principal(),
    )
    no_permission_close = get_langchain_tool(no_permission_catalog, "close_work_order_demo")
    blocked_result = json.loads(
        no_permission_close.invoke(
            {"business_id": "WO-1001", "close_reason": "现场已处理完成，申请关闭工单"}
        )
    )
    assert blocked_result["status"] == "block"
    assert "MISSING_TOOL_EXECUTE_PERMISSION" in blocked_result["matched_rules"]

    # StructuredTool 使用原项目 Pydantic args_model，因此参数不合法会在执行前失败。
    try:
        close_work_order.invoke({"business_id": "WO-1001"})
    except ValidationError:
        pass
    else:
        raise AssertionError("StructuredTool 未复用项目的 Pydantic 参数校验")

    assert {tool.name for tool in privileged_catalog.tools} >= {
        "get_session_status",
        "close_work_order_demo",
    }
    print("DAY31_LANGCHAIN_TOOL_SMOKE_OK")
    print("high_risk=require_confirm missing_permission=block args_schema=pydantic")


if __name__ == "__main__":
    main()

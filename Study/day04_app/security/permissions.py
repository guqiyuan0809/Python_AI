"""权限常量与角色权限映射，类似 Spring Security GrantedAuthority。"""

PERMISSION_ALL = "*"
PERMISSION_AI_INVOKE = "ai:invoke"
PERMISSION_AI_OPS_READ = "ai:ops:read"
PERMISSION_AI_SECURITY_AUDIT_READ = "ai:security:audit:read"
PERMISSION_AI_EVAL_RUN = "ai:eval:run"
PERMISSION_AI_PROMPT_PUBLISH = "ai:prompt:publish"
PERMISSION_AI_TASK_OPERATE = "ai:task:operate"
PERMISSION_KNOWLEDGE_READ = "knowledge:read"
PERMISSION_KNOWLEDGE_WRITE = "knowledge:write"
PERMISSION_TOOL_EXECUTE = "tool:execute"

# Java 透传的 AI 权限只能使用这份白名单，不能让任意请求头制造新权限。
KNOWN_AI_PERMISSIONS = frozenset(
    {
        PERMISSION_AI_INVOKE,
        PERMISSION_AI_OPS_READ,
        PERMISSION_AI_SECURITY_AUDIT_READ,
        PERMISSION_AI_EVAL_RUN,
        PERMISSION_AI_PROMPT_PUBLISH,
        PERMISSION_AI_TASK_OPERATE,
        PERMISSION_KNOWLEDGE_READ,
        PERMISSION_KNOWLEDGE_WRITE,
        PERMISSION_TOOL_EXECUTE,
    }
)


ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "viewer": frozenset(
        {
            PERMISSION_AI_INVOKE,
            PERMISSION_KNOWLEDGE_READ,
        }
    ),
    "operator": frozenset(
        {
            PERMISSION_AI_INVOKE,
            PERMISSION_AI_OPS_READ,
            PERMISSION_AI_EVAL_RUN,
            PERMISSION_AI_TASK_OPERATE,
            PERMISSION_KNOWLEDGE_READ,
            PERMISSION_KNOWLEDGE_WRITE,
            PERMISSION_TOOL_EXECUTE,
        }
    ),
    "admin": frozenset({PERMISSION_ALL}),
    "system": frozenset({PERMISSION_ALL}),
}

KNOWN_AI_ROLES = frozenset(ROLE_PERMISSIONS)


def permissions_for_roles(roles: tuple[str, ...]) -> frozenset[str]:
    permissions: set[str] = set()
    for role in roles:
        permissions.update(ROLE_PERMISSIONS.get(role, ()))
    return frozenset(permissions)

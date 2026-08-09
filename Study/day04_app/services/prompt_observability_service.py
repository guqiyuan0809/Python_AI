"""Prompt 身份观测工具。

日志只保存不可变身份和模板哈希，不保存本次请求的用户内容、上下文或完整消息正文。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptIdentity:
    """一次模型调用使用的 Prompt 身份快照。"""

    prompt_id: str | None
    prompt_name: str
    prompt_version: str
    prompt_template_hash: str
    prompt_source: str

    def as_call_log_fields(self) -> dict[str, str | None]:
        """转换为统一日志服务可直接展开的字段。"""
        return {
            "prompt_id": self.prompt_id,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "prompt_template_hash": self.prompt_template_hash,
        }


def build_prompt_template_hash(system_prompt: str, user_prompt_template: str) -> str:
    """对稳定模板计算 SHA-256；业务输入不参与计算，保证同一版本哈希稳定。"""
    canonical = f"system\n{system_prompt.strip()}\nuser\n{user_prompt_template.strip()}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def render_prompt_template(template: str, **variables: object) -> str:
    """校验并一次性替换占位符，避免业务输入中的占位符文本被二次替换。"""
    missing = [name for name in variables if f"{{{name}}}" not in template]
    if missing:
        raise RuntimeError(f"Prompt 模板缺少必需占位符：{','.join(sorted(missing))}")
    template_variables = set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", template))
    unknown = template_variables - set(variables)
    if unknown:
        raise RuntimeError(f"Prompt 模板包含未声明占位符：{','.join(sorted(unknown))}")
    names_pattern = "|".join(re.escape(name) for name in variables)
    pattern = re.compile(r"\{(" + names_pattern + r")\}")
    values = {name: str(value) for name, value in variables.items()}
    return pattern.sub(lambda match: values[match.group(1)], template)


def build_code_prompt_identity(
    *,
    prompt_name: str,
    prompt_version: str,
    system_prompt: str,
    user_prompt_template: str,
) -> PromptIdentity:
    return PromptIdentity(
        prompt_id=None,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        prompt_template_hash=build_prompt_template_hash(system_prompt, user_prompt_template),
        prompt_source="code",
    )


def build_registry_prompt_identity(prompt) -> PromptIdentity:
    """从 ai_prompt_version ORM 快照创建身份，记录实际 active 版本而非兼容常量。"""
    return PromptIdentity(
        prompt_id=str(prompt.prompt_id),
        prompt_name=prompt.prompt_name,
        prompt_version=prompt.prompt_version,
        prompt_template_hash=build_prompt_template_hash(
            prompt.system_prompt,
            prompt.user_prompt_template,
        ),
        prompt_source="database",
    )

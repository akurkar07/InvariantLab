"""AST helpers for enforcing task source import boundaries."""

import ast


def forbidden_imports(source: str, prefixes: tuple[str, ...]) -> list[tuple[int, str]]:
    """Return imports whose module names match any trusted prefix."""
    violations: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules = (
                [node.module]
                if any(
                    node.module == prefix or node.module.startswith(f"{prefix}.")
                    for prefix in prefixes
                )
                else [f"{node.module}.{alias.name}" for alias in node.names]
            )
        else:
            continue
        violations.extend(
            (node.lineno, module)
            for module in modules
            if any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes)
        )
    return violations

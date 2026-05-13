import ast
import operator

from langchain.tools import tool


ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        op = ALLOWED_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {node.op}")
        return op(safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = ALLOWED_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {node.op}")
        return op(safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {node}")


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression. Supports +, -, *, /, ** operators."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

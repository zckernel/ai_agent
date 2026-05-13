from langchain.tools import tool


@tool
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions.
    """

    try:
        result = eval(expression)

        return str(result)

    except Exception as e:
        return str(e)
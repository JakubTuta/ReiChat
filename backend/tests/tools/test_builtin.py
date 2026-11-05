from __future__ import annotations

import pytest

from app.tools.builtin import (
    BUILTIN_TOOLS_BY_NAME,
    UnsafeExpressionError,
    calculator,
    current_datetime,
    evaluate_arithmetic,
)


def test_evaluate_arithmetic_basic_operators():
    assert evaluate_arithmetic("2 + 2") == 4
    assert evaluate_arithmetic("10 - 3") == 7
    assert evaluate_arithmetic("4 * 5") == 20
    assert evaluate_arithmetic("10 / 4") == 2.5
    assert evaluate_arithmetic("10 // 4") == 2
    assert evaluate_arithmetic("10 % 3") == 1
    assert evaluate_arithmetic("2 ** 8") == 256


def test_evaluate_arithmetic_operator_precedence_and_parens():
    assert evaluate_arithmetic("2 + 2 * (3 - 1)") == 6


def test_evaluate_arithmetic_unary_minus():
    assert evaluate_arithmetic("-5 + 3") == -2


def test_evaluate_arithmetic_rejects_function_calls():
    with pytest.raises((UnsafeExpressionError, SyntaxError)):
        evaluate_arithmetic("__import__('os').system('echo hi')")


def test_evaluate_arithmetic_rejects_names():
    with pytest.raises(UnsafeExpressionError):
        evaluate_arithmetic("x + 1")


def test_evaluate_arithmetic_rejects_booleans_as_numbers():
    # bool is an int subclass in Python — explicitly excluded so "True + 1"
    # doesn't quietly evaluate instead of failing loudly.
    with pytest.raises(UnsafeExpressionError):
        evaluate_arithmetic("True + 1")


@pytest.mark.parametrize("expression", ["9**9**9**9", "2**100000", "(10**1000)**1000", "-(7**99999)"])
def test_evaluate_arithmetic_rejects_huge_integer_results_without_computing_them(expression):
    with pytest.raises(UnsafeExpressionError):
        evaluate_arithmetic(expression)


def test_evaluate_arithmetic_still_allows_large_but_reasonable_powers():
    assert evaluate_arithmetic("2 ** 64") == 18446744073709551616


def test_evaluate_arithmetic_rejects_overlong_expression():
    with pytest.raises(UnsafeExpressionError):
        evaluate_arithmetic("1+" * 200 + "1")


def test_calculator_tool_reports_float_overflow_as_error():
    assert calculator.invoke({"expression": "10.0 ** 400"}).startswith("Error:")


def test_calculator_tool_returns_string_result():
    assert calculator.invoke({"expression": "2 + 2"}) == "4"


def test_calculator_tool_returns_error_string_on_bad_expression():
    result = calculator.invoke({"expression": "1 / 0"})
    assert result.startswith("Error:")


def test_calculator_tool_returns_error_string_on_unsafe_expression():
    result = calculator.invoke({"expression": "__import__('os')"})
    assert result.startswith("Error:")


def test_current_datetime_tool_returns_utc_formatted_string():
    result = current_datetime.invoke({})
    assert result.endswith("UTC")
    assert len(result) == len("2026-08-10 12:00:00 UTC")


def test_builtin_tools_by_name_contains_both_tools():
    assert set(BUILTIN_TOOLS_BY_NAME) == {"calculator", "current_datetime"}


async def test_tools_can_be_awaited_the_way_the_chat_socket_calls_them():
    assert await calculator.ainvoke({"expression": "6 * 7"}) == "42"
    assert (await current_datetime.ainvoke({})).endswith("UTC")

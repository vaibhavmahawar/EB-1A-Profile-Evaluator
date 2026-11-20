"""Conservative recovery of JSON objects returned by text-generating providers."""

from __future__ import annotations

import json
from typing import Any


def _strip_code_fence(content: str) -> str:
    value = content.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return value


def _remove_trailing_commas(content: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(content):
        character = content[index]
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            output.append(character)
            index += 1
            continue
        if character == ",":
            lookahead = index + 1
            while lookahead < len(content) and content[lookahead].isspace():
                lookahead += 1
            if lookahead < len(content) and content[lookahead] in "}]":
                index += 1
                continue
        output.append(character)
        index += 1
    return "".join(output)


def parse_json_object(content: str) -> dict[str, Any]:
    """Extract and parse one JSON object without inventing missing data."""

    if not content or not content.strip():
        raise ValueError("Provider returned empty JSON content")
    cleaned = _strip_code_fence(content)
    decoder = json.JSONDecoder()
    for candidate in (cleaned, _remove_trailing_commas(cleaned)):
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                raise ValueError("Provider JSON response must be an object")
            return value
    raise ValueError("Provider response did not contain a valid JSON object")


def parse_json_payload(content: str) -> dict[str, Any] | list[Any]:
    """Parse a provider object or array without changing its contents."""

    if not content or not content.strip():
        raise ValueError("Provider returned empty JSON content")
    cleaned = _strip_code_fence(content)
    decoder = json.JSONDecoder()
    for candidate in (cleaned, _remove_trailing_commas(cleaned)):
        for index, character in enumerate(candidate):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, (dict, list)):
                return value
    raise ValueError("Provider response did not contain a valid JSON object or array")

"""Conservative recovery of JSON objects returned by text-generating providers."""

from __future__ import annotations

import json
import re
from typing import Any


def _strip_code_fence(content: str) -> str:
    value = content.strip().lstrip("\ufeff")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", value, flags=re.IGNORECASE | re.DOTALL)
    return fenced.group(1).strip() if fenced else value


def _candidate_contents(content: str) -> tuple[str, ...]:
    """Return conservative parsing candidates without modifying JSON values."""

    cleaned = _strip_code_fence(content)
    repaired = _remove_trailing_commas(cleaned)
    return (cleaned,) if repaired == cleaned else (cleaned, repaired)


def _decode_first_value(candidate: str):
    """Decode the outermost JSON value, avoiding accidental nested arrays."""

    starts = [index for index in (candidate.find("{"), candidate.find("[")) if index >= 0]
    if not starts:
        raise ValueError("Provider response did not contain JSON")
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(candidate[min(starts):])
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
    for candidate in _candidate_contents(content):
        try:
            value = _decode_first_value(candidate)
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
    for candidate in _candidate_contents(content):
        try:
            value = _decode_first_value(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    raise ValueError("Provider response did not contain a valid JSON object or array")

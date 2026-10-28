import json
import re
from typing import Any

from .parsing import Parse, ParameterSchema

COMPLETE_NUMBER = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z"
)

NUMBER_PREFIX = re.compile(
    r"-?(?:0|[1-9][0-9]*)?(?:\.[0-9]*)?\Z"
)


def generate_json_number(
    model: Any,
    context_ids: list[int],
    number_token_ids: list[int],
    stop_token_id: int,
    max_tokens: int = 30,
) -> list[int]:
    """Generate one JSON number using constrained decoding."""

    generated_ids: list[int] = []
    number_text = ""

    token_texts = {
        token_id: model.decode([token_id])
        for token_id in number_token_ids
    }

    stop_text = model.decode([stop_token_id])

    if stop_text not in (",", "}"):
        raise ValueError("Expected comma or closing brace")

    for _ in range(max_tokens):

        logits = model.get_logits_from_input_ids(
            context_ids + generated_ids
        )

        valid_ids = []

        # Find tokens that preserve the JSON number grammar
        for token_id, token_text in token_texts.items():

            if not token_text:
                continue

            candidate = number_text + token_text

            if NUMBER_PREFIX.fullmatch(candidate):
                valid_ids.append(token_id)

        # Stop only when number is complete
        if COMPLETE_NUMBER.fullmatch(number_text):
            valid_ids.append(stop_token_id)

        if not valid_ids:
            raise ValueError("No valid number tokens available")

        next_token_id = max(
            valid_ids,
            key=lambda token_id: logits[token_id]
        )

        # Model decided that the number is finished
        if next_token_id == stop_token_id:
            return generated_ids

        generated_ids.append(next_token_id)
        number_text += token_texts[next_token_id]

        # print("Generated:", repr(number_text))
        # print("Expected stop:", repr(model.decode([stop_token_id])))

    raise ValueError("Number exceeded maximum token limit")


def _string_piece_transition(
    piece: str,
    state: str,
    delimiter: str,
) -> tuple[str, bool, bool] | None:
    """Validate a full decoded token inside a JSON string.

    Returns (new_state, finished, delimiter_consumed) or None.
    The state tracks whether a JSON escape sequence is in progress.
    """
    position = 0
    while position < len(piece):
        char = piece[position]

        if state.startswith("u") and state[1:].isdigit():
            remaining = int(state[1:])
            if char not in "0123456789abcdefABCDEF":
                return None
            state = "normal" if remaining == 1 else f"u{remaining - 1}"
        elif state == "escape":
            if char in '\"\\/bfnrt':
                state = "normal"
            elif char == "u":
                state = "u4"
            else:
                return None
        elif char == "\\":
            state = "escape"
        elif char == '\"':
            suffix = piece[position + 1:]
            if suffix == "":
                return ("normal", True, False)
            if suffix == delimiter:
                return ("normal", True, True)
            return None
        elif ord(char) < 32:
            return None
        position += 1
    return (state, False, False)


def generate_json_string(
    model: Any,
    context_ids: list[int],
    valid_string_ids: list[int],
    stop_id: int,
    max_tokens: int = 50,
) -> tuple[list[int], bool]:
    """Generate a JSON string with escaped text and combined endings."""
    quote_id = model.encode('"')[0].tolist()[0]
    delimiter = model.decode([stop_id])
    if delimiter not in (",", "}"):
        raise ValueError("String delimiter must be ',' or '}'")

    # Supply punctuation and escape tokens excluded by the text filter.
    syntax_texts = (
        '"', '"' + delimiter,
        "\\", "\\\\", '\\"', "\\/",
        "\\n", "\\r", "\\t", "\\b", "\\f", "\\u",
    )
    syntax_pieces: dict[int, str] = {}
    for piece in syntax_texts:
        token_ids = model.encode(piece)[0].tolist()
        if len(token_ids) == 1 and model.decode(token_ids) == piece:
            syntax_pieces[token_ids[0]] = piece

    # Python adds the opening quote; Qwen chooses the contents.
    generated_ids = [quote_id]
    state = "normal"
    special_candidates = list(syntax_pieces)

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(context_ids + generated_ids)
        candidates: list[tuple[int, tuple[str, bool, bool]]] = []

        # Allow ordinary text outside an escape sequence.
        if state == "normal":
            candidates.extend(
                (token_id, ("normal", False, False))
                for token_id in valid_string_ids
            )
        else:
            for token_id in valid_string_ids:
                piece = model.decode([token_id])
                result = _string_piece_transition(piece, state, delimiter)
                if result is not None:
                    candidates.append((token_id, result))

        for token_id in special_candidates:
            transition = _string_piece_transition(
                syntax_pieces[token_id], state, delimiter
            )
            if transition is not None:
                candidates.append((token_id, transition))

        if not candidates:
            raise ValueError("No valid token can continue the JSON string")

        next_id, (next_state, finished, consumed) = max(
            candidates, key=lambda item: logits[item[0]]
        )
        generated_ids.append(next_id)
        if finished:
            return generated_ids, consumed
        state = next_state

    raise ValueError("String exceeded maximum token limit")


def get_valid_string_tokens(
    model: Any,
    vocab_size: int = 151936,
) -> list[int]:
    """Find tokens that can safely appear inside a basic JSON string."""

    valid_ids = []

    for token_id in range(vocab_size):
        text = model.decode([token_id])

        if not text:
            continue

        # Reject quotes, backslashes and control characters
        if '"' in text or "\\" in text:
            continue

        if any(ord(char) < 32 for char in text):
            continue

        # Reject replacement characters and obvious special tokens
        if "\ufffd" in text or text.startswith("<|"):
            continue

        valid_ids.append(token_id)

    return valid_ids


def get_valid_number_token(
        vocab_size: int,
        model: Any
) -> list[int]:
    token_ids: list[int] = []
    for token_id in range(vocab_size):
        token_text = model.decode([token_id])

        if token_text.isdigit():
            token_ids.append(token_id)

    return token_ids


def get_number_tokens(model: Any) -> list[int]:
    """Get token IDs that can compose JSON numbers."""

    characters = "0123456789-."
    result = []

    for character in characters:
        token_ids = model.encode(character)[0].tolist()

        if len(token_ids) == 1:
            result.append(token_ids[0])

    return result


def preserve_literal_span(
    value: str, user_prompt: str, threshold: float = 0.92
) -> str:
    """Restore original characters for a high-confidence LLM-proposed span.

    This preserves likely copied spans; it does not extract regex arguments.
    A genuinely generated/transformed value should remain unchanged.
    """
    from difflib import SequenceMatcher

    candidate = value.strip()
    if not candidate or len(candidate) < 2:
        return value
    if candidate in user_prompt:
        return candidate

    normalized = candidate.casefold().replace("\\", "/")
    max_difference = max(3, int(len(candidate) * 0.2))
    best_score = 0.0
    best_span = None
    for start in range(len(user_prompt)):
        minimum = max(start + 1, start + len(candidate) - max_difference)
        maximum = min(
            len(user_prompt), start + len(candidate) + max_difference
        )
        for end in range(minimum, maximum + 1):
            span = user_prompt[start:end]
            if not span or span[0].isspace() or span[-1].isspace():
                continue
            comparison = span.casefold().replace("\\", "/")
            score = SequenceMatcher(None, normalized, comparison).ratio()
            if score > best_score:
                best_score, best_span = score, span

    if best_score >= threshold and best_span is not None:
        return best_span
    return value


def generate_parameters(
    model: Any,
    parameter: ParameterSchema,
    user_prompt: str,
    definition: Parse,
    function: str,
    cached_string_ids: list[int] | None = None,
) -> list[int]:

    parameter_prompt = (
        "Extract the input arguments for the selected function.\n"
        "Do not execute the function.\n"
        "For literal strings, preserve exact punctuation, case, and slashes.\n"
        "Escape internal quotation marks and backslashes in JSON.\n"
        f"Function: {function}\n"
        f"Description: {definition.description}\n"
        f"Parameters: {json.dumps(parameter)}\n"
        f"User request: {user_prompt}\n"
    )

    if function == "fn_substitute_string_with_regex":
        parameter_prompt += (
            "Parameter meanings:\n"
            "- source_string: the original complete text.\n"
            "- regex: the pattern for the text to replace.\n"
            "For literal word replacements, use the word itself.\n"
            "Do not join occurrences using .* or repeat the pattern.\n"
            "- replacement: the replacement text.\n"
        )

    parameter_prompt += (
        "Return only the JSON arguments.\n"
        "Arguments: "
    )

    parameter_input_ids = model.encode(parameter_prompt)[0].tolist()

    number_ids = get_number_tokens(model)

    string_ids = (
        cached_string_ids
        if cached_string_ids is not None
        else get_valid_string_tokens(model)
    )

    generated_ids = [model.encode("{")[0].tolist()[0]]

    comma_id = model.encode(",")[0].tolist()[0]

    end_id = model.encode("}")[0].tolist()[0]

    keys = list(parameter.keys())

    delimiter_consumed = False

    for length in range(len(parameter)):

        if length >= 1 and not delimiter_consumed:
            generated_ids.append(comma_id)

        delimiter_consumed = False

        key = keys[length]

        generated_ids.extend(
            model.encode(json.dumps(key) + ":")[0].tolist()
        )

        context_ids = parameter_input_ids + generated_ids

        if length == len(keys) - 1:
            stop_id = end_id
        else:
            stop_id = comma_id

        parameter_type = parameter[key]
        parameters_ids = []
        if parameter_type["type"] in ["float", "number", "integer"]:
            parameters_ids = generate_json_number(
                model,
                context_ids,
                number_ids,
                stop_id,
            )
        elif parameter_type["type"] in ["string", "character"]:
            parameters_ids, delimiter_consumed = generate_json_string(
                model,
                context_ids,
                string_ids,
                stop_id,
                50,
            )
        else:
            raise ValueError(f"The type of {key} you wrote is not valid")

        generated_ids.extend(parameters_ids)

        # print(f"{key} =", model.decode(parameters_ids))

    if not delimiter_consumed:
        generated_ids.append(end_id)

    return generated_ids

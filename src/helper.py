from typing import List, Any,Dict
import json
import re


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

        print("Generated:", repr(number_text))
        print("Expected stop:", repr(model.decode([stop_token_id])))

    raise ValueError("Number exceeded maximum token limit")

def  generate_json_string(
                model,
                context_ids,
            ) -> list[int]:
    print("ok")
    return None

def get_valid_number_token(
        vocab_size: float,
        model: Any
) -> List[float]:
    token_ids = []
    for token_id in range(vocab_size):
        token_text = model.decode(token_id)

        if token_text.isdigit():
            token_ids.append(token_id)

    return token_ids


def generate_positive_integer(
    model,
    context_ids: list[int],
    digit_ids: list[int],
    stop_token_id: int,
    max_digits: int = 20,
) -> tuple[list[int], int]:
    """Generate a positive integer followed by its delimiter."""
    number_ids = []

    for _ in range(max_digits):
        logits = model.get_logits_from_input_ids(
            context_ids + number_ids
        )

        allowed = digit_ids.copy()

        if number_ids:
            allowed.append(stop_token_id)

        next_id = max(
            allowed,
            key=lambda token_id: logits[token_id]
        )

        if next_id == stop_token_id:
            return number_ids, stop_token_id

        number_ids.append(next_id)

    raise ValueError("Number generation exceeded max_digits")



def get_number_tokens(model: Any) -> list[int]:
    """Get token IDs that can compose JSON numbers."""

    characters = "0123456789-."
    result = []

    for character in characters:
        token_ids = model.encode(character)[0].tolist()

        if len(token_ids) == 1:
            result.append(token_ids[0])

    return result


def generate_parameters(
        model: Any,
        parameter: Dict[str | Dict[str | int]] | Any | None,
        user_prompt: str,
        definition: str,
        function: str,
        ) -> str:

    parameter_prompt = (
        "Extract the input arguments for the selected function.\n"
        "Do not calculate or execute the function.\n"
        "Copy the numerical values from the user request.\n"
        f"Function: {function}\n"
        f"Description: {definition.description}\n"
        f"Parameters: {json.dumps(parameter)}\n"
        f"User request: {user_prompt}\n"
        "Return only the JSON arguments.\n"
        "Arguments: "
    )

    parameter_input_ids = model.encode(
        parameter_prompt
        )[0].tolist()

    valid_ids = get_number_tokens(model)

    generated_ids = [model.encode("{")[0].tolist()[0]]

    comma_id = model.encode(",")[0].tolist()[0]

    end_id = model.encode("}")[0].tolist()[0]

    keys = list(parameter.keys())

    for length in range(len(parameter)):

        if length >= 1:
            generated_ids.append(comma_id)

        key = keys[length]

        generated_ids.extend(
            model.encode(json.dumps(key) + ":")[0].tolist()
        )

        context_ids = parameter_input_ids + generated_ids

        if length == len(keys) - 1:
            stop_id = end_id
        else:
            stop_id = comma_id

        parameter_type = str(parameter[key])
        print(parameter_type)

        if parameter_type in ['float', 'number', 'int', 'integer']:
            number_a_ids = generate_json_number(
                model,
                context_ids,
                valid_ids,
                stop_id,
            )
        elif parameter_type in ['string', 'character', 'str']:
            value = generate_json_string(
                model,
                context_ids,
            )
        else:
            raise ValueError(f"The type of {key} you wrote is not valid")

        generated_ids.extend(number_a_ids)


        print(f"{key} =", model.decode(number_a_ids))


    generated_ids.append(end_id)

    return generated_ids

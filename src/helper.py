from typing import List, Any
import json
import re
from heapq import nlargest


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



# def generate_json_string(
#     model: Any,
#     context_ids: list[int],
#     valid_string_ids: list[int],
#     max_tokens: int = 50,
# ) -> list[int]:
#     """Generate a simple quoted JSON string with constrained decoding."""

#     quote_ids = model.encode('"')[0].tolist()

#     if len(quote_ids) != 1:
#         raise ValueError("Expected a single quote token")

#     quote_id = quote_ids[0]

#     # Python adds the opening quote
#     generated_ids = [quote_id]

#     for _ in range(max_tokens):
#         logits = model.get_logits_from_input_ids(
#             context_ids + generated_ids
#         )

#         if model.decode(generated_ids) == '"cat':
#             top_ids = nlargest(
#                 15,
#                 range(len(logits)),
#                 key=lambda i: logits[i]
#             )

#             print("Top tokens after cat:")

#             for token_id in top_ids:
#                 print(
#                     repr(model.decode([token_id])),
#                     round(logits[token_id], 2),
#                     "allowed:", token_id in valid_string_ids
#                             or token_id == quote_id
#                 )


#         # Qwen may generate text or close the string
#         allowed_ids = valid_string_ids + [quote_id]

#         next_id = max(
#             allowed_ids,
#             key=lambda token_id: logits[token_id]
#         )

#         if next_id == quote_id:
#             # The model has finished the string
#             generated_ids.append(quote_id)
#             return generated_ids

#         generated_ids.append(next_id)
#         print("Generated tokens:", generated_ids)
#         print("Generated text:", repr(model.decode(generated_ids)))
#         print("Expected closing token:", repr(model.decode([quote_id])))

#     raise ValueError("String exceeded maximum token limit")

def generate_json_string(
    model,
    context_ids: list[int],
    valid_string_ids: list[int],
    stop_id: int,
    max_tokens: int = 50,
) -> tuple[list[int], bool]:

    quote_id = model.encode('"')[0].tolist()[0]
    stop_text = model.decode([stop_id])

    # A token such as '",' or '"}'
    combined_text = '"' + stop_text
    combined_ids = model.encode(combined_text)[0].tolist()

    combined_id = None

    if (
        len(combined_ids) == 1
        and model.decode(combined_ids) == combined_text
    ):
        combined_id = combined_ids[0]

    generated_ids = [quote_id]

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            context_ids + generated_ids
        )

        allowed_ids = valid_string_ids + [quote_id]

        # Allow closing quote + delimiter as one token
        if combined_id is not None:
            allowed_ids.append(combined_id)

        next_id = max(
            allowed_ids,
            key=lambda i: logits[i]
        )

        if next_id == quote_id:
            generated_ids.append(quote_id)
            return generated_ids, False

        if next_id == combined_id:
            generated_ids.append(combined_id)
            return generated_ids, True

        generated_ids.append(next_id)

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
        vocab_size: float,
        model: Any
) -> List[float]:
    token_ids = []
    for token_id in range(vocab_size):
        token_text = model.decode(token_id)

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


def generate_parameters(
        model: Any,
        parameter: dict | Any | None,
        user_prompt: str,
        definition: str,
        function: str,
        ) -> str:

    parameter_prompt = (
        "Extract the input arguments for the selected function.\n"
        "Do not execute the function.\n"
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


    # if function == "fn_substitute_string_with_regex":
    #     parameter_input_ids.extend(model.encode(regex_prompt)[0].tolist())
    # else:
    #     parameter_input_ids.extend(model.encode("Argumrents:")[0].tolist())


    number_ids = get_number_tokens(model)

    string_ids = get_valid_string_tokens(model)

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
        if parameter_type["type"] in ['float', 'number', 'integer']:
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

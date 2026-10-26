from typing import List, Any,Dict
import math


def function_description(prompt: str, data: List[str] | None) -> str:
    llm_prompt = (
                "You are a function calling assistant.\n"
                "Select the function that best matches the user's request.\n\n"
                "Available functions:\n"
            )

    for func in data:
        llm_prompt += (
        f"{func.name}: {func.description}\n"
        )

    llm_prompt += (
        f"\nUser request: {prompt}\n"
        "Return the matching function call as JSON.\n"
        "Function call: "
        )

    return llm_prompt

def choosing_function_gd(
        model: Any,
        func_ids: Dict[str],
        input_ids: List[int],
        generated_ids: List
        ) -> List[int]:
    selected_func_ids = []

    while True:

        if any(
                selected_func_ids == ids
                for ids in func_ids.values()
            ):
                break

        candidates = []

        for func, ids in func_ids.items():
            if ids[:len(selected_func_ids)] == selected_func_ids:
                candidates.append((func, ids))

        valid_token_ids = []

        for func, ids in candidates:
            position = len(selected_func_ids)

            if position < len(ids):
                valid_token_ids.append(ids[position])

        valid_token_ids = list(set(valid_token_ids))

        if not valid_token_ids:
            raise ValueError("No valid function tokens available")

        logits = model.get_logits_from_input_ids(
            input_ids + generated_ids
        )

        next_token_id = max(
            valid_token_ids,
            key=lambda i: logits[i]
        )

        generated_ids.append(next_token_id)
        selected_func_ids.append(next_token_id)

    return selected_func_ids

def choosing_function_LSE(
        model: Any,
        fn_names: List[str],
        input_ids: List[str],
        prefix_id: List[str]
        ) -> List[int]:
    function_score = {}

    for func in fn_names:
        func_ids = model.encode(func)[0].tolist()

        context_ids = input_ids + prefix_id

        score = score_of_function(model,context_ids, func_ids)

        function_score[func] = score

    return max(function_score, key=function_score.get)

def score_of_function(
        model: Any,
        context_ids: List[int],
        candidates_ids: List[int]
        ) -> float:
    score = 0.0
    current_ids = context_ids.copy()
    for token in candidates_ids:
        logits = model.get_logits_from_input_ids(current_ids)

        max_logit = max(logits)

        log_sum_exp =  max_logit + math.log(sum(math.exp(x - max_logit) for x in logits)) 

        token_log_prob =logits[token] - log_sum_exp 

        score += token_log_prob
        # print(f"for token :{repr(model.decode([token]))}, the score is :{score}")

        current_ids.append(token)

    return score

def build_parameter_prompt(
        user_prompt: str,
        function_name: str,
        parameters: Dict[str | Dict[str | int]] | Any
    ) -> str:
    text = (
        "Extract the arguments required by the selected function.\n"
        f"Function: {function_name}\n"
        "Parameters:\n"
    )
    for name, info in parameters.items():
        text += f"- {name}: {info['type']}\n"

    text += (
        f"\nUser request: {user_prompt}\n"
        "Return only the parameter values in JSON.\n"
        "Parameters: {"
    )

    return text




def get_valid_number_tokens(
        parameters: Dict[str | Dict[str | int]] | Any | None,
        vocabulary: str,
        model: Any,
        prompt: str
        ) -> list[int]:
    prmpt = (
        "Extract the arguments required by the selected function.\n"
    )
    prmpt += f"Select the {len(parameters)} parameter "
    parameter = model.encode(prmpt)[0].tolist()
    output = model.decode([parameter])

    return parameter


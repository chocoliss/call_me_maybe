import json
from pathlib import Path

from llm_sdk import Small_LLM_Model

from .parsing import parse, prompts_callinng
from .prompt import function_description, choosing_function_LSE
from .helper import (
    generate_parameters,
    get_valid_string_tokens,
    preserve_literal_span,
)


def main() -> None:
    parsed = parse()
    if parsed is None:
        raise SystemExit("Unable to load function definitions")
    data, fn_names = parsed

    model = Small_LLM_Model()

    prompts = prompts_callinng()
    string_ids = get_valid_string_tokens(model)

    results = []

    for user_prompt in prompts:
        llm_prompt = function_description(user_prompt, data)
        input_ids = model.encode(llm_prompt)[0].tolist()
        prefix_id = model.encode('{"name":"')[0].tolist()

        function = choosing_function_LSE(
            model, fn_names, input_ids, prefix_id
        )

        definition = next(
            element
            for element in data
            if element.name == function
        )

        parameter = definition.parameters or {}

        generated_param_ids = generate_parameters(
            model, parameter, user_prompt, definition, function,
            cached_string_ids=string_ids,
        )

        parameters_text = model.decode(generated_param_ids)
        parameters_result = json.loads(parameters_text)

        # Keep numeric Python types consistent with the declared schema.
        for key, info in parameter.items():
            expected_type = info["type"]
            value = parameters_result[key]
            if expected_type in ("number", "float"):
                if type(value) not in (int, float):
                    raise ValueError(f"{key} must be numeric")
                parameters_result[key] = float(value)
            elif expected_type == "integer":
                if type(value) is not int:
                    raise ValueError(f"{key} must be an integer")
            elif expected_type in ("string", "character"):
                if not isinstance(value, str):
                    raise ValueError(f"{key} must be a string")
                # Preserve verbatim user text when Qwen is already close.
                # Do not force changes to generated values like regexes.
                if key not in ("regex",):
                    parameters_result[key] = preserve_literal_span(
                        value, user_prompt
                    )

        # Verify required parameter names.
        if set(parameters_result.keys()) != set(parameter.keys()):
            raise ValueError(
                f"Invalid parameters for {function}"
            )

        final_result = {
            "prompt": user_prompt,
            "name": function,
            "parameters": parameters_result,
        }

        results.append(final_result)
        print(f"Processed: {user_prompt}")

    # Write all results to the required output file.
    output_path = Path("data/output/function_calling_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved {len(results)} results to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        exit(1)
    exit(0)

import json
from .parsing import parse, prompts_callinng
from llm_sdk import Small_LLM_Model
from .prompt import function_description, choosing_function_LSE
from .helper import generate_parameters
from pathlib import Path

def main():
    data, fn_names = parse()
    if data == None:
        exit(1)

    model = Small_LLM_Model()

    prompts = prompts_callinng()

    results = []


    for user_prompt in prompts:

    # user_prompt = "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"

        llm_prompt = function_description(user_prompt, data)

        input_ids = model.encode(llm_prompt)[0].tolist()

        prefix_id = model.encode('{"name":"')[0].tolist()

        function = choosing_function_LSE(model, fn_names, input_ids, prefix_id)
        # print(function)

        generated_ids = (
            prefix_id + 
            model.encode(function)[0].tolist() +
            model.encode('",')[0].tolist()
        )
        # print(model.decode(generated_ids))
        definition = next(
            element
            for element in data
            if element.name == function
            )

        parameter = definition.parameters
        # print("Parameter:", parameter.keys(), parameter.values())

        generated_param_ids = generate_parameters(model, parameter, user_prompt, definition, function)

        # generated_param_ids.append(model.encode("}")[0].tolist()[0])

        parameters_text = model.decode(generated_param_ids)

        parameters_result = json.loads(parameters_text)

# Verify required parameter names
        if set(parameters_result.keys()) != set(parameter.keys()):
            raise ValueError(
                f"Invalid parameters for {function}"
            )

        final_result = {
            "prompt": user_prompt,
            "name": function,
            "parameters": parameters_result
        }

        results.append(final_result)

        print(f"Processed: {user_prompt}")

    # Write all results to the required output file
    output_path = Path("data/output/function_calling_results.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nSaved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()

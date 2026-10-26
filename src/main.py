import json
from parsing import parse
from llm_sdk import Small_LLM_Model
from prompt import function_description, choosing_function_LSE
from helper import generate_parameters
from typing import Any, List, Dict

def main():
    data, fn_names = parse()
    if data == None:
        exit(1)

    model = Small_LLM_Model()

    user_prompt = "What is the sum of 1337.34 and 42?"

    llm_prompt = function_description(user_prompt, data)

    input_ids = model.encode(llm_prompt)[0].tolist()

    prefix_id = model.encode('{"name":"')[0].tolist()

    function = choosing_function_LSE(model, fn_names, input_ids, prefix_id)
    print(function)

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
    print("Parameter:", parameter.keys(), parameter.values())

    generated_param_ids = generate_parameters(model, parameter, user_prompt, definition, function)

    # generated_param_ids.append(model.encode("}")[0].tolist()[0])

    parameters_text = model.decode(generated_param_ids)

    parameters_result = json.loads(parameters_text)

    final_result = {
        "prompt": user_prompt,
        "name": function,
        "parameters": parameters_result
    }

    print(json.dumps(final_result, indent=2))


if __name__ == "__main__":
    main()

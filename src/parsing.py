import json
from typing import List, TypeAlias

from pydantic import BaseModel, TypeAdapter, ValidationError


ParameterSchema: TypeAlias = dict[str, dict[str, str | int]]


class Parse(BaseModel):
    name: str
    description: str
    parameters: ParameterSchema | None
    returns: dict


def parse() -> tuple[list[Parse], list[str]] | None:
    names: list[str] = []
    adapter = TypeAdapter(list[Parse])

    try:
        with open(
            "moulinette/data/input/functions_definition.json", 'r'
        ) as file:
            data = json.load(file)

        items = adapter.validate_python(data)
    except ValidationError as e:
        print(e)
    except Exception as e:
        print(e)
        return None
    else:
        try:
            for dicts in items:
                for k, v in dicts:
                    if k not in [
                        'name', 'description', 'parameters', 'returns'
                    ]:
                        raise ValueError(
                            "The keys in the dictionnary did not match."
                            "\nname, description,parameters,returns"
                        )
                    if k == 'name':
                        c = v[0]
                        if c.isnumeric():
                            raise ValueError(
                                "The name of the function is not correct"
                            )
                        names.append(v)
        except ValueError as e:
            print(f"Error : {e}")
        else:
            print("done")
            return items, names
    return None


def prompts_callinng() -> List[str]:
    prompts = list()
    with open(
        "moulinette/data/input/function_calling_tests.json", 'r'
    ) as f:
        inputs: List[dict[str, str]] = json.load(f)
    for dic in inputs:
        for value in dic.values():
            prompts.append(value)

    return prompts

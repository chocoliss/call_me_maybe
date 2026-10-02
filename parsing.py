from pydantic import BaseModel, TypeAdapter, ValidationError
import json


class Parse(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str, str | int]] | None
    returns: dict

def parse():
    adapter = TypeAdapter(list[Parse])

    try:
        with open("data/input/functions_definition.json", 'r') as file:
            data = json.load(file)

        items = adapter.validate_python(data)

    except ValidationError as e:
        print(e)
    except Exception as e:
        print(e)
        return None
    else:
        print("done")
        return data
from pathlib import Path
from pydantic import BaseModel, TypeAdapter, ValidationError
import json

class Param(BaseModel):
    name: dict[dict[str, str | int], dict[str, str | int]]

class Parse(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str,str | int]] | Param
    returns: dict

adapter = TypeAdapter(list[Parse])

try:
    with open("data/input/functions_definition.json", 'r') as file:
        data = json.load(file)

    items = adapter.validate_python(data)

except ValidationError as e:
    print(e)
except Exception as e:
    print(e)
else:
    print("done")
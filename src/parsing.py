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
        try:
            names = ['fn_add_numbers', 'fn_greet',
                    'fn_reverse_string', 'fn_get_square_root',
                    'fn_substitute_string_with_regex']
            for dicts in items:
                for d in dicts:
                    print(d)
                    if d not in ['name', 'description', 'parameters', 'returns']:
                        raise ValueError("The keys in the dictionnary did not match."
                            "\nname, description,parameters,returns")
                    if d == 'name':
                        c: str = dicts[d][0]
                        if c.isnumeric():
                            raise ValueError("The name of the function is not correct")
        except ValueError as e:
            print(f"Error : {e}")
        else:
            print("done")
            return items
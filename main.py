import json

def parsing(filepath: str) -> list[dict[str]]:
    try:
        filepath = str(filepath)
        with open(filepath, 'r') as file:
            f = json.load(file)
    except Exception as e:
        raise ValueError(e)
    else:
        return f

def main():
    try:
        f = parsing("data/input/functions_definition.json")
    except Exception as e:
        print(e)
        exit(1)
    else:
        for dicts in f:
            for d in dicts:
                print(f"{d} -> {dicts[d]}")
                if d not in ['name', 'description', 'parameters', 'returns']:
                    raise ValueError("The keys in the dictionnary did not match."
                        "\nname, description,parameters,returns")
                if d == 'name':
                    if dicts[d] not in ['fn_add_numbers', 'fn_greet',
                        'fn_reverse_string', 'fn_get_square_root', 'fn_substitute_string_with_regex']:
                        raise ValueError("The name of the function is not correct")
                              
    # print("Hello from call-me-maybe!")


if __name__ == "__main__":
    main()

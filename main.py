import json
from parsing import parse

def main():
    data = parse()
    if data == None:
        exit(1)
    try:
        for dicts in data:
            for d in dicts:
                if d not in ['name', 'description', 'parameters', 'returns']:
                    raise ValueError("The keys in the dictionnary did not match."
                        "\nname, description,parameters,returns")
                if d == 'name':
                    if dicts[d] not in ['fn_add_numbers', 'fn_greet',
                        'fn_reverse_string', 'fn_get_square_root',
                        'fn_substitute_string_with_regex']:
                        raise ValueError("The name of the function is not correct")
    except Exception as e:
        print(e)
        exit(1)
    # else:



if __name__ == "__main__":
    main()

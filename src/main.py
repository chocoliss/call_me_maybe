import json
from parsing import parse
from llm_sdk import Small_LLM_Model
def main():
    data = parse()
    if data == None:
        exit(1)

    model = Small_LLM_Model()

    prompt = "The capital of France is"
    input_ids = model.encode(prompt)[0].tolist()

    logits = model.get_logits_from_input_ids(input_ids)

    # Candidate token IDs
    candidate_ids = []

    for word in [" Paris", " London", " Berlin"]:
        ids = model.encode(word)[0].tolist()

        print(ids)

        if len(ids) == 1:
            candidate_ids.append(ids[0])

    if not candidate_ids:
        raise ValueError("No single-token candidates found")

    # Select the highest-scoring allowed token
    next_token_id = max(
        candidate_ids,
        key=lambda token_id: logits[token_id]
    )

    print(logits)

    print(logits[next_token_id])

    print("Selected token:", model.decode([next_token_id]))

    path = model.get_path_to_vocab_file()

    print(path)



if __name__ == "__main__":
    main()

# Suggested fixes for private grading

Replace your src/helper.py and src/__main__.py with the files in this archive (keep backup copies first).

Updates:
- Converts `number` / `float` fields to Python floats after JSON parsing, keeps `integer` exact.
- Handles JSON string escape sequences and closing quote-comma / quote-brace tokens.
- Adds conservative prompt-alignment for near-verbatim strings: mitigates leading whitespace, accidental smart punctuation, lost quotes and slash direction. This is heuristic postprocessing, not a guarantee of semantic accuracy.
- Computes allowed string tokens once per run.
- Handles parse() returning None without a tuple unpacking error.

Caveats:
- `\uXXXX` decoding is handled at the syntax level if the tokenizer selects appropriate pieces, but tokenizer vocabulary support varies.
- String escape candidate list only includes single-token encodings of common JSON escapes; full token-vocabulary escape discovery would be more comprehensive.
- Model inference cannot be tested in this environment with the private dataset. Run the private grader locally.
- The project input/output schema and CLI should be checked separately.

import json

problematic_file_path = "/mnt/wangjingxiong/think_on_graph/data/processed/cwq_train/path_data.json"
error_line_number = 578313  # 1-indexed based on your error message
error_column_number = 82   # 1-indexed based on your error message

try:
    with open(problematic_file_path, 'r', encoding='utf-8') as f:
        for current_line_idx, line_content in enumerate(f):
            if current_line_idx == error_line_number - 1:
                print(f"--- Examining Line {error_line_number} ---")
                print(f"Full line content (first 200 chars): {line_content[:200].strip()}") # Print a snippet of the line

                if len(line_content) >= error_column_number:
                    # Extract character at the exact column (0-indexed for string)
                    char_in_question = line_content[error_column_number - 1]
                    char_ord_value = ord(char_in_question)

                    print(f"Character at column {error_column_number}: '{char_in_question}'")
                    print(f"Unicode code point (decimal): {char_ord_value}")
                    print(f"Unicode code point (hex): 0x{char_ord_value:04x}")

                    if char_ord_value < 32 and char_in_question not in ['\t', '\n', '\r', '\f', '\b']: # Standard escapable control chars
                        print("Verdict: This IS an unescaped control character disallowed in JSON strings.")
                    elif char_ord_value == 127: # DEL character
                        print("Verdict: This IS the DEL control character, disallowed in JSON strings.")
                    else:
                        print("Verdict: This character's code point itself is not a typical control character (0-31, 127) usually causing this error directly, but the JSON parser still flagged an issue here. There might be encoding issues or other subtle problems if this is not a control character.")
                    
                    # Print context around the character
                    start_context = max(0, error_column_number - 1 - 10) # 10 chars before
                    end_context = min(len(line_content), error_column_number - 1 + 11) # 10 chars after (char itself + 10)
                    context_snippet = line_content[start_context:end_context]
                    print(f"Context snippet around column {error_column_number}: '{context_snippet}'")
                    # For a clearer view of non-printables in context
                    print(f"Context snippet (repr): {repr(context_snippet)}")

                else:
                    print(f"Error: Line {error_line_number} is shorter than {error_column_number} columns.")
                
                # Try to parse just this line to see if the error is isolated here
                try:
                    json.loads(f'{{"key":"{line_content.strip()}"}}') # Test if the line content is valid in a string
                    print(f"Test: Line {error_line_number} content seems okay when embedded in a simple JSON string if properly escaped.")
                except json.JSONDecodeError as e_line:
                    print(f"Test: Line {error_line_number} content itself causes JSON error when embedded: {e_line}")
                break
        else:
            print(f"Error: File does not have {error_line_number} lines.")

except FileNotFoundError:
    print(f"Error: File not found at '{problematic_file_path}'")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
"""
Usage: python3 format_json.py /path/to/data.json
"""
import sys
import json

def escape_control_chars_in_strings(s: str) -> str:
    out = []
    in_string = False
    escape = False
    for ch in s:
        if ch == '"' and not escape:
            in_string = not in_string
            out.append(ch)
            continue

        if in_string and not escape:
            if ch == '\n':
                out.append('\\n')
                continue
            if ch == '\r':
                out.append('\\r')
                continue
            if ch == '\t':
                out.append('\\t')
                continue

        if ch == '\\' and not escape:
            escape = True
            out.append(ch)
            continue
        else:
            out.append(ch)
            escape = False

    return ''.join(out)


def main():
    if len(sys.argv) < 2:
        print("Usage: format_json.py <file.json>")
        sys.exit(2)

    path = sys.argv[1]
    txt = open(path, 'r', encoding='utf-8').read()

    fixed = escape_control_chars_in_strings(txt)

    try:
        obj = json.loads(fixed)
    except Exception as e:
        print("Failed to parse JSON after escaping control characters:", e)
        sys.exit(1)

    pretty = json.dumps(obj, indent=2, ensure_ascii=False)
    open(path, 'w', encoding='utf-8').write(pretty + "\n")
    print(f"Formatted and wrote JSON to {path}")


if __name__ == '__main__':
    main()

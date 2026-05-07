from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parent
APP_FILE = ROOT / "app.py"
OUT_DIR = ROOT / "public"
OUT_FILE = OUT_DIR / "index.html"


def extract_app_html() -> str:
    tree = ast.parse(APP_FILE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "APP_HTML":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, str):
                        raise TypeError("APP_HTML is not a string")
                    return value
    raise RuntimeError("Could not find APP_HTML in app.py")


def main() -> None:
    app_html = extract_app_html()
    OUT_DIR.mkdir(exist_ok=True)
    OUT_FILE.write_text(
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "  <title>Three-Body Simulation</title>\n"
        "</head>\n"
        "<body>\n"
        f"{app_html}\n"
        "</body>\n"
        "</html>\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

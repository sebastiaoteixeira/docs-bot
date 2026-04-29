import os
import subprocess
import json
import re
from google import genai
from google.genai import types
from pathlib import Path

MAX_LINES = 250
MAX_CHARS = 2500


def run_command(command):
    try:
        return subprocess.check_output(
            command, shell=True, text=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e.output}")
        raise


def _truncate_lines(lines: list[str], from_line: int) -> tuple[list[str], bool]:
    selected = lines[from_line - 1 : from_line - 1 + MAX_LINES]
    line_truncated = from_line - 1 + MAX_LINES < len(lines)
    return selected, line_truncated


def _truncate_chars(lines: list[str], from_line: int) -> tuple[str, bool, int, int]:
    result_lines = []
    total_chars = 0
    char_truncated = False
    for line in lines:
        if total_chars + len(line) > MAX_CHARS:
            char_truncated = True
            break
        result_lines.append(line)
        total_chars += len(line)

    end_line = from_line + len(result_lines) - 1
    content = "".join(result_lines)
    return content, char_truncated, from_line, end_line


def read_doc_file(path: str) -> str:
    """Reads the content of a specific documentation file.
    Use this to inspect existing docs before proposing updates.
    """
    p = Path(path)
    if "docs" not in p.parts:
        return "Error: Access denied. You can only read files within the 'docs/' directory."

    if not p.exists():
        return f"Error: File '{path}' does not exist."

    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {str(e)}"


def read_src_file(path: str, from_line: int = 1) -> dict:
    """Reads a range of lines from a source file under src/.

    Args:
        path: File path relative to repo root (e.g. "src/yaml_api_generator/ir.py").
        from_line: 1-based start line number. Defaults to 1.

    Returns:
        A dict with the file content (line-numbered), the line range returned,
        and truncation flags.
    """
    p = Path(path)
    if not p.exists():
        return {"error": f"File '{path}' does not exist."}

    if "src" not in p.parts:
        return {"error": "Access denied. You can only read files under src/."}

    try:
        all_lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    except Exception as e:
        return {"error": str(e)}

    if from_line < 1:
        from_line = 1

    total_lines = len(all_lines)
    selected, line_truncated = _truncate_lines(all_lines, from_line)
    content, char_truncated, actual_start, actual_end = _truncate_chars(
        selected, from_line
    )

    numbered = "".join(
        f"{i}: {line}" for i, line in zip(range(actual_start, actual_end + 1), selected)
    )

    return {
        "file": path,
        "lines": f"{actual_start}-{actual_end}",
        "total_lines": total_lines,
        "content": numbered,
        "truncated_by_lines": line_truncated,
        "truncated_by_chars": char_truncated,
    }


def grep_src(pattern: str, path: str = "src/", from_line: int = 1) -> dict:
    """Searches for a regex pattern in source files under src/.

    Args:
        pattern: Regular expression to search for.
        path: Directory or file to search in. Defaults to "src/".
        from_line: 1-based index into the match results to start from (for pagination).

    Returns:
        A dict with matching file:line entries, truncated to 250 lines and 2500 chars.
    """
    p = Path(path)
    if not p.exists():
        return {"error": f"Path '{path}' does not exist."}

    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return {"error": f"Invalid regex: {e}"}

    matches: list[str] = []
    files_to_search = [p] if p.is_file() else sorted(p.rglob("*.py"))

    for fp in files_to_search:
        if not fp.is_file():
            continue
        try:
            lines = fp.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if compiled.search(line):
                matches.append(f"{fp}:{i}: {line.rstrip()}")
                if len(matches) >= MAX_LINES * 5:
                    break
        if len(matches) >= MAX_LINES * 5:
            break

    total_matches = len(matches)
    from_line = max(1, from_line)
    selected = matches[from_line - 1 : from_line - 1 + MAX_LINES]
    line_truncated = from_line - 1 + MAX_LINES < total_matches

    content, char_truncated, actual_start, actual_end = _truncate_chars(
        selected, from_line
    )

    return {
        "pattern": pattern,
        "path": path,
        "matches_shown": f"{actual_start}-{actual_end}",
        "total_matches": total_matches,
        "content": content,
        "truncated_by_lines": line_truncated,
        "truncated_by_chars": char_truncated,
    }


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        return

    # 1. Get the list of changed files and the tree
    try:
        # Get the tree to give AI the 'map'
        docs_tree = run_command("find docs -maxdepth 5 -not -path '*/.*'")

        # Get the diff to show what changed in code
        diff = run_command("git diff HEAD~1 HEAD -- 'src/**'")
        if not diff.strip():
            print("No source code changes detected. Skipping docs update.")
            return
    except Exception as e:
        print(f"Could not get git context: {e}")
        return

    # 2. Configure Gemini with Tools
    client = genai.Client(api_key=api_key)

    tools = [read_doc_file, read_src_file, grep_src]

    custom_prompt = os.getenv("DOCS_BOT_PROMPT", "").strip()

    base_prompt = f"""
    You are an expert Technical Writer Agent for the 'yaml-api-generator' framework.
    
    DOCS DIRECTORY STRUCTURE:
    {docs_tree}
    
    TASK:
    Update the documentation to reflect the provided code changes.
    
    AGENTIC WORKFLOW:
    1. Review the Code Diff.
    2. Identify which documentation files (from the STRUCTURE above) are affected.
    3. Use the 'read_doc_file' tool to read existing docs before updating them.
    4. Use the 'read_src_file' tool to inspect source code for accurate documentation.
    5. Use the 'grep_src' tool to search for patterns across the source code.
    6. Propose new files if the feature is entirely undocumented.
    
    CRITICAL INSTRUCTIONS:
    - Maintain existing tone, style, and formatting (Fumadocs/Markdown).
    - Suggest updates to 'meta.json' or index files if you add new pages.
    - Return your final answer ONLY as a JSON object.
    
    RESPONSE FORMAT:
    Return a valid JSON object:
    {{
      "files": [
        {{"path": "docs/path/to/file.md", "content": "Full updated content"}}
      ],
      "pr_title": "Short descriptive title",
      "pr_body": "Detailed summary of changes"
    }}
    """

    system_prompt = (
        f"{base_prompt}\n\nDOC ORGANIZATION GUIDELINES:\n{custom_prompt}"
        if custom_prompt
        else base_prompt
    )

    TOOL_MAP = {
        "read_doc_file": read_doc_file,
        "read_src_file": read_src_file,
        "grep_src": grep_src,
    }

    def run_turn(contents):
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=tools,
            ),
        )
        return response

    def execute_function_call(fc):
        fn = TOOL_MAP.get(fc.name)
        if fn is None:
            return f"Error: unknown function {fc.name}"
        args = dict(fc.args) if fc.args else {}
        print(f"  Tool call: {fc.name}({args})")
        return fn(**args)

    try:
        print(f"Agent starting analysis using {model_name}...")
        contents = [f"CODE DIFF:\n{diff}"]

        for _ in range(10):
            response = run_turn(contents)

            if response.text:
                break

            parts = response.candidates[0].content.parts
            tool_call_parts = []
            tool_response_parts = []

            for part in parts:
                if part.function_call:
                    result = execute_function_call(part.function_call)
                    tool_call_parts.append(part)
                    tool_response_parts.append(
                        types.Part.from_function_response(
                            name=part.function_call.name,
                            response={"result": result},
                        )
                    )

            contents.append(types.Content(role="model", parts=tool_call_parts))
            contents.append(types.Content(role="user", parts=tool_response_parts))

        raw_text = response.text
        if raw_text is None:
            print("Model returned no text after multi-turn loop.")
            return

        content = raw_text.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]

        data = json.loads(content.strip())
    except Exception as e:
        print(f"Agent failed to produce valid JSON: {e}")
        print(f"Raw response: {raw_text if 'raw_text' in dir() else 'No response'}")
        return

    # 4. Apply changes
    for file_data in data.get("files", []):
        path = Path(file_data["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file_data["content"], encoding="utf-8")
        print(f"Successfully processed: {path}")

    # 5. Export for GitHub Actions
    if "GITHUB_ENV" in os.environ:
        with open(os.environ["GITHUB_ENV"], "a") as f:
            f.write(f"AI_PR_TITLE={data.get('pr_title', 'docs: automatic update')}\n")
            f.write("AI_PR_BODY<<EOF\n")
            f.write(
                f"{data.get('pr_body', 'Documentation update triggered by code changes.')}\n"
            )
            f.write("EOF\n")


if __name__ == "__main__":
    main()

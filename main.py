import os
import subprocess
import json
from google import genai
from google.genai import types
from pathlib import Path


def run_command(command):
    try:
        return subprocess.check_output(
            command, shell=True, text=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e.output}")
        raise


def read_doc_file(path: str) -> str:
    """Reads the content of a specific documentation file.
    Use this to inspect existing docs before proposing updates.
    """
    p = Path(path)
    # Security check: don't allow reading outside docs/
    if "docs" not in p.parts:
        return "Error: Access denied. You can only read files within the 'docs/' directory."

    if not p.exists():
        return f"Error: File '{path}' does not exist."

    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {str(e)}"


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

    # We provide the read_doc_file function as a tool
    tools = [read_doc_file]

    system_prompt = f"""
    You are an expert Technical Writer Agent for the 'yaml-api-generator' framework.
    
    DOCS DIRECTORY STRUCTURE:
    {docs_tree}
    
    TASK:
    Update the documentation to reflect the provided code changes.
    
    AGENTIC WORKFLOW:
    1. Review the Code Diff.
    2. Identify which documentation files (from the STRUCTURE above) are affected.
    3. Use the 'read_doc_file' tool to read those files so you can update them correctly.
    4. Propose new files if the feature is entirely undocumented.
    
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

    TOOL_MAP = {"read_doc_file": read_doc_file}

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

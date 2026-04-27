import os
import subprocess
import json
from google import genai
from pathlib import Path

def run_command(command):
    try:
        return subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e.output}")
        raise

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        return

    # 1. Get the diff from the caller repository
    try:
        diff = run_command("git diff HEAD~1 HEAD -- 'src/**'")
        if not diff.strip():
            print("No source code changes detected in 'src/'. Skipping docs update.")
            return
    except Exception as e:
        print(f"Could not get git diff: {e}")
        return

    # 2. Configure Modern Gemini Client
    client = genai.Client(api_key=api_key)

    # 3. Prepare the Prompt
    system_prompt = """
    You are a specialized technical writer for the 'yaml-api-generator' framework.
    This framework generates REST APIs (FastAPI/DRF) and Frontends (React) from YAML specs.
    
    TASK:
    Analyze the provided 'git diff' of code changes and update the corresponding documentation in '/docs'.
    
    GUIDELINES:
    1. If a new CLI flag is added to 'cli.py', update CLI reference.
    2. If a new YAML key is added to the parser/schema, update the Reference section.
    3. If frontend logic changes, update the UI/Frontend documentation.
    4. Maintain the existing tone, formatting, and Fumadocs structure.
    
    RESPONSE FORMAT:
    Return your response ONLY as a valid JSON object:
    {
      "files": [
        {"path": "docs/path/to/file.md", "content": "Full markdown content"}
      ],
      "pr_title": "Short descriptive title (max 70 chars)",
      "pr_body": "Markdown summary of documentation changes"
    }
    """

    prompt = f"{system_prompt}\n\nGIT DIFF:\n{diff}"

    # 4. Generate Content
    try:
        print(f"Generating docs using model: {model_name}")
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        
        content = response.text.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        
        data = json.loads(content.strip())
    except Exception as e:
        print(f"Error parsing AI response: {e}")
        return

    # 5. Apply changes to the filesystem
    for file_data in data.get("files", []):
        path = Path(file_data["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file_data["content"])
        print(f"Updated: {path}")

    # 6. Export PR info to GitHub Environment
    if 'GITHUB_ENV' in os.environ:
        with open(os.environ['GITHUB_ENV'], 'a') as f:
            f.write(f"AI_PR_TITLE={data.get('pr_title', 'docs: automatic update')}\n")
            f.write("AI_PR_BODY<<EOF\n")
            f.write(f"{data.get('pr_body', 'Updated documentation based on code changes.')}\n")
            f.write("EOF\n")

if __name__ == "__main__":
    main()

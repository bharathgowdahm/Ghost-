import os, ast, subprocess
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from 【entity-github¦canonical_name=GitHub】 import 【entity-Github¦canonical_name=GitHub】

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
MODEL = os.getenv("MODEL", "gpt-4o-mini")

def analyze_repo(repo_url):
    if not repo_url:
        return "Paste a 【entity-GitHub¦canonical_name=GitHub】 URL", "", ""

    # Clone shallow
    name = repo_url.rstrip('/').split('/')[-1]
    path = f"/tmp/{name}"
    subprocess.run(["rm", "-rf", path])
    subprocess.run(["git", "clone", "--depth", "1", repo_url, path])

    # Code stats
    files = []
    for root, _, filenames in os.walk(path):
        for f in filenames:
            if f.endswith(('.py','.js','.ts','.go')):
                files.append(os.path.join(root, f))

    tree = f"Total code files: {len(files)}\n"
    sample_code = ""
    if files:
        with open(files[0], 'r', errors='ignore') as fp:
            sample_code = fp.read()[:4000]

    # AI Analysis
    if client:
        prompt = f"""You are a senior GitHub code reviewer.
        Analyze this repo: {repo_url}
        Sample file content:
        {sample_code}
        Give:
        1. What this project does (2 lines)
        2. Tech Stack
        3. 3 Security / Quality issues
        4. How to improve to get 1k stars
        """
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":prompt}]
        )
        ai_report = resp.choices[0].message.content
    else:
        ai_report = "⚠️ Add OPENAI_API_KEY in.env to get real AI report (mock mode active)"

    # Auto README
    readme_prompt = f"Generate a professional GitHub README.md for {repo_url} based on: {sample_code[:2000]}. Include badges, features, install, usage."
    if client:
        r2 = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":readme_prompt}])
        readme = r2.choices[0].message.content
    else:
        readme = f"# {name}\nAuto-generated README (add API key for full version)"

    return tree, ai_report, readme

with gr.Blocks(theme=gr.themes.Glass(), title="GHOST") as demo:
    gr.Markdown("# 👻 GHOST - GitHub Intelligence OS\nPaste any GitHub repo → Get AI report + Pro README in 10s")
    url = gr.Textbox(label="GitHub Repo URL", placeholder="https://github.com/psf/requests")
    btn = gr.Button("Analyze 🔍", variant="primary")
    with gr.Row():
        stats = gr.Textbox(label="Stats")
        report = gr.Markdown(label="AI Intelligence Report")
    readme_out = gr.Markdown(label="Generated README.md")
    btn.click(analyze_repo, inputs=url, outputs=[stats, report, readme_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")

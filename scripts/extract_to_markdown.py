import json
import os

def extract_code(ipynb_path, out_txt_path):
    if not os.path.exists(ipynb_path):
        print(f"File not found: {ipynb_path}")
        return
    
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cells = data.get('cells', [])
    code_idx = 1
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, 'w', encoding='utf-8') as out_f:
        out_f.write(f"# Extracted code from: {os.path.basename(ipynb_path)}\n\n")
        for cell in cells:
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                code_text = "".join(source)
                if code_text.strip():
                    out_f.write(f"## Code Cell {code_idx}\n")
                    out_f.write("```python\n")
                    out_f.write(code_text)
                    if not code_text.endswith('\n'):
                        out_f.write('\n')
                    out_f.write("```\n\n")
                    code_idx += 1
    print(f"Extracted {code_idx-1} cells to {out_txt_path}")

if __name__ == "__main__":
    extract_code(
        r"c:\Users\PC\Desktop\ZenAI\PHASE 2\FLUX+QWEN\FluxInfer.ipynb",
        r"C:\Users\PC\.gemini\antigravity-ide\brain\66e08116-8a29-49bb-ba51-b178fd7eece7\scratch\flux_code.md"
    )
    extract_code(
        r"c:\Users\PC\Desktop\ZenAI\PHASE 2\FLUX+QWEN\QwenInfer.ipynb",
        r"C:\Users\PC\.gemini\antigravity-ide\brain\66e08116-8a29-49bb-ba51-b178fd7eece7\scratch\qwen_code.md"
    )

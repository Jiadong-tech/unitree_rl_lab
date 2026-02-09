import os
from markdown_it import MarkdownIt

# 配置路径
INPUT_FILE = "doc/Complete_Robot_Training_Guide_CN.md"
OUTPUT_FILE = "doc/Complete_Robot_Training_Guide_CN.html"

# CSS 样式 (GitHub 风格)
CSS = """
<style>
    body {
        font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
        line-height: 1.6;
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem;
        color: #24292e;
    }
    h1, h2, h3 { border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
    code { background-color: #f6f8fa; padding: 0.2em 0.4em; border-radius: 3px; }
    pre { background-color: #f6f8fa; padding: 16px; overflow: auto; border-radius: 3px; }
    pre code { background-color: transparent; padding: 0; }
    blockquote { border-left: 0.25em solid #dfe2e5; color: #6a737d; padding: 0 1em; margin: 0; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 16px; }
    table th, table td { padding: 6px 13px; border: 1px solid #dfe2e5; }
    table tr:nth-child(2n) { background-color: #f6f8fa; }
</style>
"""

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: 找不到文件 {INPUT_FILE}")
        return

    # 读取 Markdown
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        md_text = f.read()

    # 转换为 HTML
    md = MarkdownIt()
    html_content = md.render(md_text)

    # 组合完整 HTML
    full_html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>Unitree RL Lab 指导书</title>
        {CSS}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # 写入文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_html)
    
    print(f"成功生成: {OUTPUT_FILE}")
    print("请双击该 HTML 文件在浏览器中打开，然后使用 'Ctrl+P' 另存为 PDF。")

if __name__ == "__main__":
    main()

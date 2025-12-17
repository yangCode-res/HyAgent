import os
import pathlib
import base64
from mistralai import Mistral

API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL = "mistral-ocr-latest"   # 你之前用的可跑通的模型名

def encode_pdf_base64(path):
    with open(path, "rb") as f:
        return "data:application/pdf;base64," + base64.b64encode(f.read()).decode()


def ocr_from_urls(url_list):
    """返回每个 URL 的 OCR 文本"""
    results = []

    with Mistral(api_key=API_KEY) as client:
        for url in url_list:
            print("Processing:", url)

            try:
                # 判断 URL vs 本地路径
                if url.startswith("http://") or url.startswith("https://"):
                    document_payload = {
                        "document_url": url,
                        "type": "document_url"
                    }
                else:
                    b64 = encode_pdf_base64(url)
                    document_payload = {
                        "document_base64": b64,
                        "type": "document_base64"
                    }

                res = client.ocr.process(
                    model=MODEL,
                    document=document_payload
                )

                # 合并页内容
                pages = []
                for p in res.pages:
                    if getattr(p, "markdown", None):
                        pages.append(p.markdown)
                    elif getattr(p, "text", None):
                        pages.append(p.text)

                results.append("\n\n".join(pages))

            except Exception as e:
                print("Error:", e)
                results.append(None)

    return results


# ----------------------------------------------------
# 📌 你现在要的包装函数：输入 URL 列表 → 输出保存的 MD 文件路径列表
# ----------------------------------------------------
def ocr_to_md_files(url_list, save_dir="ocr_md_outputs"):
    """
    输入: url_list = [url1, url2, ...]
    输出: md_paths = ["xxx/file1.md", "xxx/file2.md", ...]
    """
    # 创建保存目录
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    md_paths = []

    # 拿到 OCR 文本
    texts = ocr_from_urls(url_list)

    for idx, text in enumerate(texts):
        if text is None:
            md_paths.append(None)
            continue

        # 生成文件名
        md_path = save_dir / f"ocr_result_{idx+1}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)

        md_paths.append(str(md_path))

    return md_paths


# ================= 示例运行 =================
if __name__ == "__main__":
    urls = [
        "https://arxiv.org/pdf/2407.08940.pdf",
        "/mnt/data/2407.08940v2.pdf"
    ]

    md_files = ocr_to_md_files(urls)
    print("\n>>> 保存的 Markdown 文件列表：")
    print(md_files)

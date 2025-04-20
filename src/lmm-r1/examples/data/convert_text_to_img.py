import fitz
def get_bbox(input_pdf):
    # 打开PDF文件
    pdf_document = fitz.open(input_pdf)
    
    
    # 遍历每一页
    page = pdf_document.load_page(0)
    
    # 获取页面的边界框
    rect = page.rect
    
    # 获取页面的所有图像和文本块
    blocks = page.get_text("blocks")
    images = page.get_image_info()
    # 找到所有内容的边界
    min_x = rect.width
    min_y = rect.height
    max_x = 0
    max_y = 0
    
    # 遍历文本块
    for block in blocks:
        x0, y0, x1, y1, _, _, _ = block
        min_x = min(min_x, x0)
        min_y = min(min_y, y0)
        max_x = max(max_x, x1)
        max_y = max(max_y, y1)
    
    # 遍历图像
    for image in images:
        x0, y0, x1, y1 = image["bbox"]
        min_x = min(min_x, x0)
        min_y = min(min_y, y0)
        max_x = max(max_x, x1)
        max_y = max(max_y, y1)
    pdf_document.close()
    #归一化
    min_x = min_x / rect.width
    min_y = min_y / rect.height
    max_x = max_x / rect.width
    max_y = max_y / rect.height
    return min_x, min_y, max_x, max_y
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import PyPDF2
from pdf2image import convert_from_path
from PIL import ImageOps
import json
from tqdm import tqdm
import random
SYSTEM_PROMPT = (
    "You are a helpful assistant good at solving math problems with step-by-step reasoning. You should "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$."
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>."
)


class TypstToImageConverter:
    def __init__(
        self,
        output_dir="output",
        font_size="16pt",
        page_size=None,
        margins=None,
        dpi=300,
    ):
        self.output_dir = Path(output_dir)
        self.font_size = font_size
        self.page_size = page_size or {"width":"10cm","height":"10cm"}
        self.margins = margins or {"x": "0.5cm", "y":"0.5cm"}
        self.dpi = dpi

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_typst_template(self, content):
        # 尺寸参数生成逻辑
        page_param = f"width:{self.page_size['width']},height:{self.page_size['height']}, margin:(x:{self.margins['x']},y:{self.margins['y']})"

        return rf"""
#import "@preview/mitex:0.2.5": *
#set page({page_param})
#set text(size: {self.font_size})

#mitext(`{content}`)
"""

    def process_question(self, content, question_id):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            typst_content = self._build_typst_template(content)
            typst_file = tmp_path / "temp.typ"
            with open(typst_file, "w", encoding="utf-8") as f:
                f.write(typst_content)
            pdf_file = tmp_path / "temp.pdf"
            try:
                subprocess.run(
                    [
                        "typst",
                        "compile",
                        str(typst_file),
                        str(pdf_file),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=10,
                )
            except subprocess.CalledProcessError as e:
                print(f"Error compiling Typst for question {question_id}: {e.stderr.decode()}")
                return False
            except subprocess.TimeoutExpired:
                print(f"Typst compilation timeout for question {question_id}")
                return False

            # 检查PDF页数
            
            if not pdf_file.exists():
                return False

            try:
                with open(pdf_file, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    if len(reader.pages) >= 2:
                        return False
            except PyPDF2.errors.PdfReadError:
                print(f"Invalid PDF generated for question {question_id}")
                return False

            # 转换为图像
            images = convert_from_path(pdf_file, dpi=self.dpi)
            for page_num, image in enumerate(images):
                output_file = self.output_dir / f"q{question_id}.jpg"
                bbox = get_bbox(pdf_file)
                width = image.width
                height = image.height
                bbox = (bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height)
                image = image.crop(bbox)
                image = ImageOps.expand(image, border=10, fill='white')
                image.save(output_file)


            return True

def process_single_question(args):
    idx, question, answer, output_dir, converter = args
    try:
        success = converter.process_question(question, idx)
        if success:
            content = [
                {
                    "type": "text",
                    "text": "Answer the question in the image."
                },
                {
                    "type": "image",
                    "image": "file://" + output_dir + f"q{idx}.jpg"
                }
            ]
            if random.random() > 0.5:
                content = content[::-1]
            message = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
            return {
                "success": True,
                "data": {
                    "message": json.dumps(message, ensure_ascii=False),
                    "question": question,
                    "answer": answer
                }
            }
        return {"success": False, "data": question}
    except Exception as e:
        print(f"Error processing question {idx}: {str(e)}")
        return {"success": False, "data": question}

if __name__ == "__main__":
    import multiprocessing as mp
    
    with open("data/deepscaler.json", 'r') as f:
        data = json.load(f)
    
    output_dir = "/root/projects/lmm-r1/data/deepscaler_img/"
    converter = TypstToImageConverter(
        output_dir=output_dir,
        font_size="16pt",
        page_size={"width": "15cm", "height": "10cm"},
        dpi=120,
    )
    
    # 准备参数列表
    process_args = []
    for idx, d in enumerate(data):
        answer = d['answer']
        if len(answer) == 0:
            continue
        if answer[0] != "$":
            answer = "$" + answer + "$"
        question = d['problem']
        question = question.replace("\\\\", "\\")
        process_args.append((idx, question, answer, output_dir, converter))
    
    # 使用进程池并行处理
    num_processes = 16
    with mp.Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_question, process_args),
            total=len(process_args),
            desc="Processing questions"
        ))
    
    # 处理结果
    final_data = []
    fail_questions = []
    for result in results:
        if result["success"]:
            final_data.append(result["data"])
        else:
            fail_questions.append(result["data"])

    print(f"Successfully processed: {len(final_data)}")
    print(f"Failed questions: {len(fail_questions)}")
    with open("data/deepscaler_img.jsonl",'w') as f:
        for d in final_data:
            f.write(json.dumps(d,ensure_ascii=False)+"\n")

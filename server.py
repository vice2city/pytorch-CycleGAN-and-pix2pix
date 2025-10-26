import sys
import torch
import base64
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import torch
from models.test_model import TestModel
from options.test_options import TestOptions
from data.base_dataset import get_transform

opt = None
model = None

app = FastAPI()

@app.on_event("startup")
def load_model_on_startup():
    global opt, model
    print("Loading model inside startup...")
    sys.argv = [sys.argv[0]]
    opt = TestOptions().parse()  # 如果 parse() 会消费 argv，这里 uvicorn 的 CLI 已经不会受影响
    opt.device = torch.device("cpu")
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.no_dropout = True
    opt.preprocess = 'scale_width'
    opt.name = 'opt2sar'

    model = TestModel(opt)
    model.setup(opt)
    model.eval()
    print("Model loaded successfully!")


class ProcessedImage(BaseModel):
    """定义一个包含文件名和图片数据的对象"""
    filename: str
    image_data: str  # Base64 编码的图片字符串

class ImagesResponse(BaseModel):
    """响应体，包含一个 ProcessedImage 对象的列表"""
    processed_images: List[ProcessedImage]

def tensor_to_base64(tensor_image):
    """将 PyTorch tensor 转换成 base64 字符串"""
    # 将 tensor 转换回 PIL Image
    # 这个转换过程可能需要根据您模型的具体输出格式进行微调
    # 以下是一个通用的示例
    tensor_image = (tensor_image.data.squeeze(0).cpu().float().numpy() + 1.0) / 2.0 * 255.0
    tensor_image = tensor_image.astype('uint8')
    if tensor_image.shape[0] == 1: # Grayscale
        image_pil = Image.fromarray(tensor_image.squeeze(0), 'L')
    else: # RGB
        image_pil = Image.fromarray(tensor_image.transpose(1, 2, 0))

    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.post("/predict", response_model=ImagesResponse)
async def predict(files: List[UploadFile] = File(...)):
    """
    接收批量图片文件，使用 CycleGAN 进行转换，并返回结果。
    """
    global model, opt
    if opt is None or model is None:
        return {"message": "Model is not loaded yet."}
    processed_results = []
    transform = None
    print(f"Received {len(files)} files for processing.")

    for file in files:
        original_filename = file.filename
        print(f"Processing image: {original_filename}")
        # 读取上传的图片
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        if transform is None:
            height, width = image.size
            opt.load_size = max(height, width, 1024)
            transform = get_transform(opt)

        # 图像预处理
        input_tensor = transform(image)

        # 模型推理
        with torch.no_grad():
            model.set_input({'A': input_tensor, 'A_paths': ''})
            model.forward()
            output_tensor = model.get_current_visuals()['fake']

        # 结果后处理
        base64_image = tensor_to_base64(output_tensor)
        processed_results.append(
            ProcessedImage(filename=original_filename, image_data=base64_image)
        )

    print("Processing complete, returning results.")
    return ImagesResponse(processed_images=processed_results)

@app.get("/")
def read_root():
    if model is None:
        return {"message": "Model is not loaded yet."}
    return {"message": "CycleGAN model server is running."}

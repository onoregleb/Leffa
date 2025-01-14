from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from tempfile import NamedTemporaryFile
import shutil
import numpy as np
from PIL import Image
from app import LeffaPredictor

app = FastAPI()

leffa_predictor = None

@app.on_event("startup")
async def startup_event():
    global leffa_predictor
    leffa_predictor = LeffaPredictor()

@app.post("/virtual-tryon/")
async def virtual_tryon(
    src_image: UploadFile = File(...),
    ref_image: UploadFile = File(...),
    control_type: str = "virtual_tryon",
    step: int = 30,
    scale: float = 2.5,
    seed: int = 42,
    vt_model_type: str = "viton_hd",
    vt_garment_type: str = "upper_body",
):
    """
    API для виртуальной примерки одежды.

    Параметры:
        - `src_image`: Изображение пользователя.
        - `ref_image`: Изображение одежды.
        - `control_type`: Тип управления (по умолчанию `virtual_tryon`).
        - `step`: Количество шагов инференса (по умолчанию 50).
        - `scale`: Коэффициент управления (по умолчанию 2.5).
        - `seed`: Сид для генерации (по умолчанию 42).
        - `vt_model_type`: Тип модели примерки (по умолчанию `viton_hd`).
        - `vt_garment_type`: Тип одежды для примерки (по умолчанию `upper_body`).

    Возвращает:
        - Сгенерированное изображение.
    """
    if leffa_predictor is None:
        raise HTTPException(status_code=500, detail="Модель не инициализирована.")

    try:
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_src_file:
            shutil.copyfileobj(src_image.file, temp_src_file)
            src_image_path = temp_src_file.name

        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_ref_file:
            shutil.copyfileobj(ref_image.file, temp_ref_file)
            ref_image_path = temp_ref_file.name

        generated_image, _, _ = leffa_predictor.leffa_predict(
            src_image_path=src_image_path,
            ref_image_path=ref_image_path,
            control_type=control_type,
            step=step,
            scale=scale,
            seed=seed,
            vt_model_type=vt_model_type,
            vt_garment_type=vt_garment_type,
        )

        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_out_file:
            output_image = Image.fromarray(generated_image)
            output_image.save(temp_out_file.name)
            output_path = temp_out_file.name

        return FileResponse(output_path, media_type="image/jpeg")

    finally:
        src_image.file.close()
        ref_image.file.close()

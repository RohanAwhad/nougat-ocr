import argparse
import base64

from fastapi import FastAPI, HTTPException
from io import BytesIO
from PIL import Image
from pydantic import BaseModel

import main

app = FastAPI()

class ImageData(BaseModel):
  image: str
  max_len: int = 100

@app.post("/upload-image/")
async def upload_image(data: ImageData):
  try:
    img_bytes = base64.b64decode(data.image)
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    text = main.generate([img], max_len=data.max_len)
    return {"result": text}
  except Exception as e:
    raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt', type=str, default=None, help='Path to model checkpoint')
  args = parser.parse_args()
  main.config, main.processor, main.encoder, main.decoder, main.device = main.get_model(args.ckpt)

  import uvicorn
  uvicorn.run(app, host="localhost", port=8000)

# Nougat-OCR

This is a simpler implementation of Meta's Nougat model that was trained to convert an image of a research paper page into markdown content. The [original repo](https://github.com/facebookresearch/nougat) doesn't run easily on a mac silicon, so I wrote my custom models inspired by transformers implementation.

The model is a VisionEncoderDecoderModel. Encoder is a Donut SwinTransformer and Decoder is a MBartDecoder.

### How to run:

1. Make sure your transformers package >= '4.38.2'
2. Run the following command to extract text from `make_a_scene.pdf`
  ```bash
  python3 main.py --ckpt "rawhad/fb-nougat-small-split" -i make_a_scene.pdf -o make_a_scene.txt
  ```

> [!NOTE]
> It can run in "cuda", "cpu" or "mps", but for now I do not use "mps" because I haven't been able to get good performance on it

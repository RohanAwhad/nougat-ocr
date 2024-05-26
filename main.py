import os
import torch

from src.models import DonutSwinEncoder, NougatDecoder
from transformers import NougatProcessor, VisionEncoderDecoderConfig

ckpt = "/Users/rohan/3_Resources/ai_models/nougat-small"
config = VisionEncoderDecoderConfig.from_pretrained(ckpt)
processor = NougatProcessor.from_pretrained(ckpt)

encoder = DonutSwinEncoder(config.encoder)
decoder = NougatDecoder(
  embed_dim=config.decoder.d_model,
  num_layers=config.decoder.decoder_layers,
  vocab_size=config.decoder.vocab_size,
  scale_embedding=config.decoder.scale_embedding,
  num_heads=config.decoder.decoder_attention_heads,
  max_len=config.decoder.max_position_embeddings,
)
print('Model loaded')

encoder.load_state_dict(torch.load(os.path.join(ckpt, 'encoder.bin')))
decoder.load_state_dict(torch.load(os.path.join(ckpt, 'decoder.bin')))
print('Weights loaded')

encoder.eval()
decoder.eval()

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('mps') if torch.backends.mps.is_available() else device

encoder.to(device)
decoder.to(device)

@torch.no_grad()
def generate(images, max_len=None):
  pixel_values = processor(images, return_tensors='pt').pixel_values.to(device)
  encoder_hidden_state = encoder(pixel_values)

  bz = encoder_hidden_state.shape[0]
  input_ids = [[0] for _ in range(bz)]
  output_ids = [[0] for _ in range(bz)]

  n_layers = config.decoder.decoder_layers
  embed_dim = config.decoder.d_model

  ks_cache = torch.empty(n_layers, bz, 0, embed_dim).to(device)
  vs_cache = torch.empty(n_layers, bz, 0, embed_dim).to(device)
  kc_cache = torch.empty(n_layers, bz, 0, embed_dim).to(device)
  vc_cache = torch.empty(n_layers, bz, 0, embed_dim).to(device)

  cache = (ks_cache, vs_cache, kc_cache, vc_cache)

  inp = torch.tensor(input_ids).to(device)
  break_flag = False
  while not break_flag:
    n_inp_tokens = len(input_ids[0])
    cross_attention_mask = torch.zeros(1, 1, n_inp_tokens, encoder_hidden_state.shape[1]).to(device)
    self_attention_mask = torch.zeros(1, 1, n_inp_tokens, n_inp_tokens).to(device)

    kv_logits, cache = decoder(
      inp,
      self_attention_mask,
      encoder_hidden_state,
      cross_attention_mask,
      *cache
    )

    next_token_ids = kv_logits[:, -1, :].argmax(dim=-1)
    break_flag = True
    for i, x in enumerate(next_token_ids):
      if x not in (0, 1, 2): break_flag = False
      output_ids[i].append(x)

    inp = next_token_ids.unsqueeze(1)
    print(processor.decode(output_ids[1][-1]), end='', flush=True)

    if max_len and len(output_ids[0]) >= max_len: break

  return output_ids


if __name__ == '__main__':
  import pdf2image
  import numpy as np
  import time
  from PIL import Image

  #image = Image.open('tmp.png').convert('RGB')
  images = pdf2image.convert_from_path('make_a_scene.pdf')
  images = [np.array(image) for image in images]
  images = [Image.fromarray(image).convert("RGB") for image in images]
  start = time.monotonic()
  output_ids = generate(images, max_len=300)
  end = time.monotonic()
  with open('make_a_scene.txt', 'w') as f:
    for x in output_ids:
      f.write(processor.decode(x, skip_special_tokens=True))
  print()
  print('='*80)
  print(f'Time taken: {end-start:.2f} seconds')
  print('='*80)

  

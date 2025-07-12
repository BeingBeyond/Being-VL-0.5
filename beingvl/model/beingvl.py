from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
import torch
from PIL import Image
import json
import numpy as np
from .vbpe import vBPE


class BeingVL:
    """Being-VL multimodal model for vision-language tasks with vBPE support."""

    def __init__(self, model_path=None, vbpe_path=None, device_map="auto"):
        """Initialize Being-VL model.
        
        Args:
            model_path: Path to trained Being-VL model
            vbpe_path: Path to vBPE tokenizer
            device_map: Device mapping for model loading
        """
        self.model_path = model_path
        self.vbpe_path = vbpe_path
        self.device_map = device_map
        self.tokenize_result = []

    def load_model(self, model_path=None, vbpe_path=None, vbpe_init_size=8192, 
                   vbpe_vocab_pad=128261, vbpe_vocab_end=136453, torch_dtype=torch.bfloat16, 
                   flash_attn=True):
        """Load model and vBPE tokenizer."""
        if model_path:
            self.model_path = model_path
        if vbpe_path:
            self.vbpe_path = vbpe_path

        if flash_attn:
            self.model = ChameleonForConditionalGeneration.from_pretrained(
                self.model_path, 
                device_map=self.device_map, 
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                )
        else:
            self.model = ChameleonForConditionalGeneration.from_pretrained(
                self.model_path, 
                device_map=self.device_map, 
                torch_dtype=torch_dtype,
                )
        self.vqgan = self.model.model
        self.processor = ChameleonProcessor.from_pretrained(self.model_path)
        print(f"Model loaded: {self.model_path} on {self.model.device}")

        self.vbpe = vBPE(vbpe_init_size, vbpe_vocab_pad, vbpe_vocab_end)
        if self.vbpe_path:
            self.vbpe.load_vocab(self.vbpe_path)

    def load_data(self, json_path, num=0):
        """Load training data from JSON file."""
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        if num > 0:
            self.json_data = json_data[:num]
        else:
            self.json_data = json_data
        print(f"Data loaded: {json_path}")

    def img2tok(self, imgs):
        """Convert images to VQ tokens."""
        img_pixels = self.processor.image_processor(
            imgs, 
            return_tensors='pt'
            )["pixel_values"].to(self.model.device, self.model.dtype)
        _, _, img_token = self.model.model.vqmodel.encode(img_pixels)
        return img_token + self.vbpe.vocab_pad

    def tok2img(self, img_token):
        """Convert VQ tokens back to image."""
        token_tensor = torch.LongTensor(img_token).unsqueeze(0) - self.vbpe.vocab_pad
        recon_pixel = self.model.model.vqmodel.decode(token_tensor.to(self.model.device))
        recon_img = self.processor.postprocess_pixel_values(recon_pixel)
        image_array = recon_img[0].detach().cpu().numpy()
        return Image.fromarray(np.transpose(image_array, (1, 2, 0)))

    def generate(self, prompts, images, max_new_tokens=200, do_sample=False):
        """Generate responses using standard Chameleon processor."""
        generated_results = []
        
        for prompt, image in zip(prompts, images):
            inputs = self.processor(prompt, images=[image], return_tensors="pt").to(self.model.device, self.model.dtype)
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
            generated_results.append(self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
        
        return generated_results

    def generate_vq(self, prompts, images, max_new_tokens=200, do_sample=False):
        """Generate responses using VQ tokens and vBPE compression."""
        generated_results = []
        image_tokens = self.img2tok(images)
        if self.vbpe_path:
            trans_tokens = [self.vbpe.trans_token(image_token.cpu().numpy()).tolist() for image_token in image_tokens]
        else:
            trans_tokens = [image_token.tolist() for image_token in image_tokens]

        for prompt, ext_tok in zip(prompts, trans_tokens):
            ext_input_ids = [self.processor.tokenizer.bos_token_id, self.model.vocabulary_mapping.boi_token_id] + \
                ext_tok + \
                [self.model.vocabulary_mapping.eoi_token_id] + self.processor.tokenizer.encode(prompt)[1:]
            ext_input_ids = torch.tensor(
                ext_input_ids, 
                dtype=torch.long
            ).unsqueeze(0).to(self.model.device)
            attention_mask = self.model._prepare_attention_mask_for_generation(
                ext_input_ids,
                torch.tensor(self.processor.tokenizer.pad_token_id, device=self.model.device, dtype=torch.long),
                torch.tensor(self.processor.tokenizer.eos_token_id, device=self.model.device, dtype=torch.long)
            )
            inputs = {
                "input_ids": ext_input_ids,
                "attention_mask": attention_mask
            }
            input_len = ext_input_ids.shape[1]
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
            generated_results.append(self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0])
        
        return generated_results

    def batch_tokenize_single_turn(self, images, data):
        """Tokenize single-turn conversations."""
        pre = [self.processor.tokenizer.bos_token_id, self.model.vocabulary_mapping.boi_token_id]
        image_tokens = self.img2tok(images)

        for i, dt in enumerate(data):
            ret = ""
            ret += '<eoss>' + dt['conversations'][0]['value'].replace('<image>', '').strip()
            ret += ' ' + dt['conversations'][1]['value'] + '</s>'
            entry = {
                'image': dt['image'],
                'original': f'<s><racm3:break><image>{ret}',
                'token_0': pre,
                'token_1': self.processor.tokenizer.encode(ret)[1:],
                'token_image': image_tokens[i].tolist(),
            }
            self.tokenize_result.append(entry)

    def batch_tokenize_multi_turn_multi_entry(self, images, data):
        """Tokenize multi-turn conversations as single entries."""
        pre = [self.processor.tokenizer.bos_token_id, self.model.vocabulary_mapping.boi_token_id]
        image_tokens = self.img2tok(images)

        for i, dt in enumerate(data):
            ret = ""
            for k, v in enumerate(dt['conversations']):
                if k == 0:
                    ret += '<eoss>' + v['value'].replace('<image>', '').strip()
                elif k % 2 == 0:
                    ret += '<reserved08706>' + v['value']
                else:
                    ret += ' ' + v['value'] + '</s>'
            entry = {
                'image': dt['image'],
                'original': f'<s><racm3:break><image>{ret}',
                'token_0': pre,
                'token_1': self.processor.tokenizer.encode(ret)[1:],
                'token_image': image_tokens[i].tolist(),
            }
            self.tokenize_result.append(entry)

    def batch_tokenize_multi_turn_single_entry(self, images, data):
        """Tokenize multi-turn conversations into separate entries."""
        pre = [self.processor.tokenizer.bos_token_id, self.model.vocabulary_mapping.boi_token_id]
        image_tokens = self.img2tok(images)

        for i, dt in enumerate(data):
            for k in range(len(dt['conversations'])):
                tmp = ""
                if k == 0:
                    tmp += '<eoss>' + dt['conversations'][k]['value'].replace('<image>', '').strip()
                    tmp += ' ' + dt['conversations'][k+1]['value'] + '</s>'
                elif k % 2 == 0:
                    tmp += '<eoss>' + dt['conversations'][k]['value']
                    tmp += ' ' + dt['conversations'][k+1]['value'] + '</s>'
                else:
                    continue
                
                entry = {
                    'image': dt['image'],
                    'original': f'<s><racm3:break><image>{tmp}',
                    'token_0': pre,
                    'token_1': self.processor.tokenizer.encode(tmp)[1:],
                    'token_image': image_tokens[i].tolist(),
                }
                self.tokenize_result.append(entry)

    def save_tokenize_result(self, save_path):
        """Save tokenized results to JSONL file."""
        with open(save_path, 'w') as f:
            for entry in self.tokenize_result:
                f.write(json.dumps(entry) + '\n')
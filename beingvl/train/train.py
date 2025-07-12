import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor, Trainer, TrainingArguments, HfArgumentParser
from torch.nn.utils.rnn import pad_sequence
import jsonlines
from pathlib import Path
from typing import Dict, Optional, Sequence, List
import random


@dataclass
class ModelArguments:
    """Model configuration arguments."""
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)  # chameleon+8k, llama2+vqgan, llama3+vqgan, etc.


@dataclass
class DataArguments:
    """Data processing arguments."""
    data_path: str = field(
        default='',
        metadata={"help": "Path to the training data."}
    )
    shuffle_data: bool = field(default=False)


@dataclass
class TrainingArguments(TrainingArguments):
    """Extended training arguments for Being-VL."""
    freeze_text_heads: bool = field(default=False)  # Freeze all layers except output heads
    init_image_heads: bool = field(default=False)   # Initialize image token heads
    train_with_vbpe: bool = field(default=False)    # Use vBPE compressed tokens


class TokenizedDataset(Dataset):
    """Dataset for tokenized multimodal training data."""
    
    def __init__(self, filepath, train_with_vbpe=False, shuffle_data=False):
        self.tokenized_data = []
        if train_with_vbpe:
            # Use vBPE compressed image tokens
            with jsonlines.open(filepath) as reader:
                for obj in reader:
                    self.tokenized_data.append(torch.tensor(obj['token_0'] + obj['token_image_new'] + obj['token_1'], dtype=torch.long))
        else:
            # Use original VQ image tokens
            with jsonlines.open(filepath) as reader:
                for obj in reader:
                    self.tokenized_data.append(torch.tensor(obj['token_0'] + obj['token_image'] + obj['token_1'], dtype=torch.long))
        if shuffle_data:
            random.shuffle(self.tokenized_data)
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx],


def collate_fn(batch):
    """Custom collate function for padding variable-length sequences."""
    batch_inputs = [item[0] for item in batch]
    batch_inputs_padded = pad_sequence(batch_inputs, batch_first=True, padding_value=-100)
    attention_masks = torch.zeros_like(batch_inputs_padded, dtype=torch.long)
    attention_masks = attention_masks.masked_fill(batch_inputs_padded != -100, 1)
    return {'input_ids': batch_inputs_padded, 'attention_mask': attention_masks, 'labels': batch_inputs_padded.clone()}


def zero_out_gradient_chameleon_8k(grad):
    """Mask gradients for chameleon+8k model to prevent training unwanted tokens."""
    grad[:4, :] = 0        # Preserve special tokens
    grad[8196:65536, :] = 0  # Mask unused vocabulary range
    return grad

def zero_out_gradient_llama2_vqgan(grad):
    """Mask gradients for llama2+vqgan model."""
    grad[32000:, :] = 0  # Only train new image tokens
    return grad

def zero_out_gradient_llama3_vqgan(grad):
    """Mask gradients for llama3+vqgan model."""
    grad[128256:, :] = 0  # Only train new image tokens
    return grad

parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
model = ChameleonForConditionalGeneration.from_pretrained(
    model_args.model_name_or_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
processor = ChameleonProcessor.from_pretrained(model_args.model_name_or_path)

# Stage 1 training: freeze all layers except output heads
if training_args.freeze_text_heads:
    for param in model.parameters():
        param.requires_grad = False
    print("Freezed all layers except the output layer.")
    for param in model.lm_head.parameters():
        if training_args.init_image_heads:
            # He initialization for new image token embeddings
            std_dev = (2.0 / 4096) ** 0.5
            if model_args.model_type == 'chameleon+8k':
                param[4:8196] = torch.randn((8196 - 4, 4096)) * std_dev
                param[65536:65536 + 8192] = torch.randn((8192, 4096)) * std_dev
                print(f"Initializing range: {4}-{8196}, {65536}-{65536 + 8192}")
            elif model_args.model_type == 'llama2+vqgan':
                param[32000:32000 + 5 + 8192] = torch.randn((32000 + 5 + 8192 - 32000, 4096)) * std_dev
                print(f"Initializing range: {32000}-{32000 + 5 + 8192}")
            elif model_args.model_type == 'llama2+vqgan+8k':
                param[32000:32000 + 5 + 8192 + 8192] = torch.randn((32000 + 5 + 8192 + 8192 - 32000, 4096)) * std_dev
                print(f"Initializing range: {32000}-{32000 + 5 + 8192 + 8192}")
            elif model_args.model_type == 'llama3+vqgan':
                param[128256:128256 + 5 + 8192] = torch.randn((128256 + 5 + 8192 - 128256, 4096)) * std_dev
                print(f"Initializing range: {128256}-{128256 + 5 + 8192}")
            elif model_args.model_type == 'llama3+vqgan+8k':
                param[128256:128256 + 5 + 8192 + 8192] = torch.randn((128256 + 5 + 8192 + 8192 - 128256, 4096)) * std_dev
                print(f"Initializing range: {128256}-{128256 + 5 + 8192 + 8192}")
            else:
                raise ValueError(f'Unknown model type: {model_args.model_type}')
            print("Initialized the image heads with He initialization.")
        param.requires_grad = True
        # Register gradient masking hooks for model-specific vocabulary ranges
        if model_args.model_type == 'chameleon+8k':
            param.register_hook(zero_out_gradient_chameleon_8k)
            print(f"Registered zero out grad hook for chameleon+8k type.")
        elif model_args.model_type.startswith('llama2'):
            param.register_hook(zero_out_gradient_llama2_vqgan)
            print(f"Registered zero out grad hook for llama2+vqgan type.")
        elif model_args.model_type.startswith('llama3'):
            param.register_hook(zero_out_gradient_llama3_vqgan)
            print(f"Registered zero out grad hook for llama3+vqgan type.")
        else:
            raise ValueError(f'Unknown model type: {model_args.model_type}')

dataset = TokenizedDataset(
    data_args.data_path, 
    training_args.train_with_vbpe,
    data_args.shuffle_data,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn
)

trainer.train()

torch.save(model.state_dict(), Path(training_args.output_dir) / 'pytorch_model.bin')
model.save_pretrained(Path(training_args.output_dir) / 'final')
processor.save_pretrained(Path(training_args.output_dir) / 'final')
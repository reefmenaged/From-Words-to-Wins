"""One frozen Q-A encoder implementation shared by both supported BERT models."""
from __future__ import annotations

from typing import List, Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from .config import DEFAULT_MAX_LENGTH
from .data import QAPair
from .runtime import resolve_device


class FrozenQAEncoder:
    """Frozen Hugging Face encoder that returns final-layer first-token Q-A vectors.

    Each Q-A pair is tokenized as two sequences. This preserves the behavior of
    both original implementations: model special tokens are inserted by the
    tokenizer and the final hidden state's position 0 representation is used.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        try:
            self.model = AutoModel.from_pretrained(model_name, dtype=dtype)
        except TypeError:  # compatibility with older Transformers releases
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)

        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.hidden_size = int(self.model.config.hidden_size)

    def encode_interviews(
        self,
        interviews: Sequence[Sequence[QAPair]],
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> List[torch.Tensor]:
        """Encode each interview as a variable-length tensor of Q-A embeddings."""
        counts = [len(interview) for interview in interviews]
        if any(count <= 0 for count in counts):
            raise ValueError("Every interview must contain at least one Q-A pair.")

        flat: List[QAPair] = [pair for interview in interviews for pair in interview]
        all_vectors: List[torch.Tensor] = []
        iterator = range(0, len(flat), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Frozen {self.model_name}: Q-A embeddings")

        with torch.inference_mode():
            for start in iterator:
                batch = flat[start : start + batch_size]
                questions = [q for q, _ in batch]
                answers = [a for _, a in batch]
                tokens = self.tokenizer(
                    questions,
                    text_pair=answers,
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                outputs = self.model(**tokens)
                cls_vectors = outputs.last_hidden_state[:, 0, :]
                all_vectors.extend(cls_vectors.float().cpu().unbind(dim=0))

        result: List[torch.Tensor] = []
        cursor = 0
        for count in counts:
            result.append(torch.stack(all_vectors[cursor : cursor + count], dim=0))
            cursor += count
        return result

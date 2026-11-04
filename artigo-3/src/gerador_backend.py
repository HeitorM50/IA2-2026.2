"""Backend do gerador: modelo aberto local via Hugging Face transformers (#52).

Decisão da #46 (04/11/2026): Qwen2.5-3B-Instruct, revisão fixada por hash em
`config.py`, decodificação gulosa. Gulosa porque a temperatura do protocolo é 0
e porque, com a mesma GPU e as mesmas versões, ela repete a saída — o que a
amostragem não garante.

O `transformers` e o `torch` só são importados em `construir_gerador`, para que
os testes e as etapas sem geração rodem sem eles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import CONFIG, GeracaoConfig
from src.contratos import SaidaGerador


@dataclass
class GeradorHF:
    """Envolve tokenizador e modelo causais com chat template."""

    nome: str
    revisao: str
    tokenizer: Any
    model: Any

    def gerar(self, prompt: str, *, temperatura: float, max_tokens: int) -> SaidaGerador:
        if temperatura != 0.0:
            raise ValueError("O protocolo usa decodificação gulosa; temperatura deve ser 0.")
        mensagens = [{"role": "user", "content": prompt}]
        # return_dict explícito: o padrão mudou entre versões do transformers.
        lote = self.tokenizer.apply_chat_template(
            mensagens, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        entrada = lote["input_ids"].to(self.model.device)
        saida = self.model.generate(
            entrada,
            attention_mask=lote["attention_mask"].to(self.model.device),
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        novos = saida[0][entrada.shape[-1]:]
        texto = self.tokenizer.decode(novos, skip_special_tokens=True).strip()
        return SaidaGerador(
            texto=texto,
            tokens_entrada=int(entrada.shape[-1]),
            tokens_saida=int(len(novos)),
        )


def construir_gerador(cfg: GeracaoConfig = CONFIG.geracao) -> GeradorHF:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(CONFIG.seed)
    usa_gpu = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(cfg.modelo_gerador, revision=cfg.revisao_gerador)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.modelo_gerador,
        revision=cfg.revisao_gerador,
        dtype=torch.float16 if usa_gpu else torch.float32,
        device_map="auto" if usa_gpu else None,
    )
    model.eval()
    return GeradorHF(nome=cfg.modelo_gerador, revisao=cfg.revisao_gerador, tokenizer=tokenizer, model=model)

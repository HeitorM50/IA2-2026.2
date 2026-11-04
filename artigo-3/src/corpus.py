"""Extração auditável do corpus das aulas 14–23 (#48).

Cada página de cada PDF vira um registro com origem inequívoca: aula, parte,
título, número da página, nome do arquivo e SHA-256 do PDF. Slides com colunas,
fórmulas e figuras costumam extrair mal, então páginas vazias ou curtas demais
são marcadas como suspeitas em vez de descartadas em silêncio.

A saída é determinística: a mesma entrada gera os mesmos arquivos byte a byte,
o que permite que o manifesto seja conferido antes da execução canônica (#53).

Uso::

    python -m src.corpus            # extrai para data/extraido/ e grava o manifesto
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import CONFIG, CorpusConfig

PARTES_VALIDAS = ("V", "VI", "VII")
VERSAO_EXTRATOR = 1


class CorpusError(RuntimeError):
    """Corpus ausente, incompleto ou com metadado inválido."""


@dataclass(frozen=True)
class ArquivoAula:
    aula: int
    caminho: Path


@dataclass(frozen=True)
class Pagina:
    aula: int
    parte: str | None
    titulo: str | None
    pagina: int  # 1-based, como aparece no leitor de PDF
    origem: str  # nome do arquivo
    sha256_pdf: str
    texto: str
    n_caracteres: int
    suspeita: str | None  # motivo, ou None quando a página parece ok


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def normalizar(texto: str) -> str:
    """Normaliza espaços sem apagar quebras de linha nem fronteiras de bloco.

    - converte espaços não separáveis e tabulações em espaço comum;
    - colapsa espaços repetidos dentro da linha e remove os das pontas;
    - colapsa três ou mais quebras seguidas em uma linha em branco.
    """
    texto = texto.replace(" ", " ").replace("\t", " ").replace("\r", "\n")
    linhas = [re.sub(r" {2,}", " ", linha).strip() for linha in texto.split("\n")]
    texto = "\n".join(linhas)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def validar_catalogo(cfg: CorpusConfig) -> None:
    for aula, entrada in cfg.catalogo.items():
        if aula not in cfg.aulas:
            raise CorpusError(f"Catálogo cita a aula {aula}, fora de {cfg.aulas}.")
        if not (isinstance(entrada, tuple) and len(entrada) == 2):
            raise CorpusError(f"Aula {aula}: catálogo deve ser (parte, título).")
        parte, titulo = entrada
        if parte not in PARTES_VALIDAS:
            raise CorpusError(
                f"Aula {aula}: parte {parte!r} inválida; use uma de {PARTES_VALIDAS}."
            )
        if not str(titulo).strip():
            raise CorpusError(f"Aula {aula}: título vazio no catálogo.")


def descobrir_pdfs(cfg: CorpusConfig) -> list[ArquivoAula]:
    """Encontra exatamente um PDF por aula esperada."""
    if not cfg.corpus_dir.is_dir():
        raise CorpusError(
            f"Diretório do corpus não encontrado: {cfg.corpus_dir}. "
            "Copie os PDFs para lá ou defina IA2_CORPUS_DIR."
        )
    padrao = re.compile(cfg.padrao_nome_arquivo)
    por_aula: dict[int, list[Path]] = {}
    sem_aula: list[str] = []
    for caminho in sorted(cfg.corpus_dir.glob("*.pdf")):
        m = padrao.search(caminho.stem)
        if not m:
            sem_aula.append(caminho.name)
            continue
        por_aula.setdefault(int(m.group(1)), []).append(caminho)

    erros: list[str] = []
    if sem_aula:
        erros.append(f"sem número de aula no nome: {sem_aula}")
    duplicadas = {a: [p.name for p in ps] for a, ps in por_aula.items() if len(ps) > 1}
    if duplicadas:
        erros.append(f"aula com mais de um PDF: {duplicadas}")
    fora = sorted(set(por_aula) - set(cfg.aulas))
    if fora:
        erros.append(f"aulas fora do recorte: {fora}")
    faltando = sorted(set(cfg.aulas) - set(por_aula))
    if faltando:
        erros.append(f"aulas sem PDF: {faltando}")
    if erros:
        raise CorpusError("Corpus inválido — " + "; ".join(erros))

    return [ArquivoAula(aula=a, caminho=por_aula[a][0]) for a in sorted(por_aula)]


def extrair_pdf(arquivo: ArquivoAula, cfg: CorpusConfig) -> list[Pagina]:
    import fitz  # PyMuPDF; importado aqui para o resto do módulo não depender dele

    parte, titulo = cfg.catalogo.get(arquivo.aula, (None, None))
    digest = sha256_arquivo(arquivo.caminho)
    paginas: list[Pagina] = []
    with fitz.open(arquivo.caminho) as doc:
        for i, page in enumerate(doc, start=1):
            # sort=True ordena os blocos por posição (cima→baixo, esquerda→direita),
            # o que reduz texto fora de ordem em slides com colunas.
            texto = normalizar(page.get_text("text", sort=True))
            n = len(texto)
            if n == 0:
                suspeita = "sem_texto"
            elif n < cfg.min_caracteres_pagina:
                suspeita = "texto_curto"
            else:
                suspeita = None
            paginas.append(
                Pagina(
                    aula=arquivo.aula,
                    parte=parte,
                    titulo=titulo,
                    pagina=i,
                    origem=arquivo.caminho.name,
                    sha256_pdf=digest,
                    texto=texto,
                    n_caracteres=n,
                    suspeita=suspeita,
                )
            )
    return paginas


def _jsonl(registros: list[dict]) -> str:
    return "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in registros
    )


def extrair_corpus(cfg: CorpusConfig = CONFIG.corpus) -> dict:
    """Extrai todas as aulas, grava um JSONL por aula e devolve o manifesto."""
    validar_catalogo(cfg)
    arquivos = descobrir_pdfs(cfg)
    cfg.extraido_dir.mkdir(parents=True, exist_ok=True)

    entradas = []
    for arquivo in arquivos:
        paginas = extrair_pdf(arquivo, cfg)
        conteudo = _jsonl([asdict(p) for p in paginas])
        destino = cfg.extraido_dir / f"aula-{arquivo.aula:02d}.jsonl"
        destino.write_text(conteudo, encoding="utf-8")
        entradas.append(
            {
                "aula": arquivo.aula,
                "origem": arquivo.caminho.name,
                "sha256_pdf": paginas[0].sha256_pdf if paginas else sha256_arquivo(arquivo.caminho),
                "n_paginas": len(paginas),
                "paginas_suspeitas": [
                    {"pagina": p.pagina, "motivo": p.suspeita}
                    for p in paginas
                    if p.suspeita
                ],
                "extraido": destino.name,
                "sha256_extraido": hashlib.sha256(conteudo.encode("utf-8")).hexdigest(),
            }
        )

    # Sem data/hora nem caminho absoluto: o manifesto precisa ser idêntico entre
    # máquinas para servir de verificação na execução canônica.
    manifesto = {
        "versao_extrator": VERSAO_EXTRATOR,
        "aulas": list(cfg.aulas),
        "catalogo_completo": set(cfg.catalogo) == set(cfg.aulas),
        "min_caracteres_pagina": cfg.min_caracteres_pagina,
        "arquivos": entradas,
    }
    cfg.manifesto_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.manifesto_path.write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifesto


def carregar_paginas(cfg: CorpusConfig = CONFIG.corpus) -> list[Pagina]:
    """Lê o corpus já extraído, na ordem aula → página."""
    paginas: list[Pagina] = []
    for caminho in sorted(cfg.extraido_dir.glob("aula-*.jsonl")):
        for linha in caminho.read_text(encoding="utf-8").splitlines():
            paginas.append(Pagina(**json.loads(linha)))
    return paginas


def main() -> int:
    try:
        manifesto = extrair_corpus()
    except CorpusError as e:
        print(f"erro: {e}", file=sys.stderr)
        return 1
    total = sum(a["n_paginas"] for a in manifesto["arquivos"])
    suspeitas = sum(len(a["paginas_suspeitas"]) for a in manifesto["arquivos"])
    print(f"{len(manifesto['arquivos'])} PDFs, {total} páginas, {suspeitas} suspeitas")
    if not manifesto["catalogo_completo"]:
        print("aviso: catálogo aula→(parte, título) incompleto em src/config.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

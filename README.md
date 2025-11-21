# Katube 2025

Pipeline de processamento de audio do YouTube para criacao de datasets TTS em portugues brasileiro.

## Status
Em desenvolvimento ativo - Prompt 00/16 concluido

## Requisitos
- Python 3.8+
- FFmpeg
- Sox
- CUDA 11.x+ (opcional, recomendado)

## Setup
```bash
# Instalar dependencias (apos PROMPT 16)
pip install -r requirements.txt

# Configurar
# 1. Editar config.py (HUGGINGFACE_TOKEN obrigatorio)
# 2. Colocar audios em audios/{video_id}/{video_id}.flac

# Executar
python run.py
```

## Estrutura
Ver `GUIA_IMPLEMENTACAO_KATUBE_2025.md` para detalhes completos.

## Licenca
MIT

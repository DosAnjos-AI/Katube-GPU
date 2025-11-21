"""
Configuracao centralizada do Katube 2025.
TODOS os parametros da pipeline devem estar aqui.
Nunca hardcode valores nos modulos - sempre referencia config.

Autor: Equipe Katube
Data: 2025-01-15
"""
from pathlib import Path

# Validacao automatica de GPU (torch opcional no setup inicial)
try:
    import torch
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU nao disponivel, usando CPU")
except ImportError:
    print("PyTorch nao instalado - sera necessario para execucao da pipeline")

# ============================================================================
# PATHS - Input/Output
# ============================================================================

# Caminho da pasta que contem as subpastas com audios
# Estrutura esperada: {AUDIO_INPUT_DIR}/video_id/*.flac
AUDIO_INPUT_DIR = Path("audios")

# Outputs (FIXO, nao configuravel)
# dataset/audio_dataset/, dataset/historico_dataset/, etc.

# ============================================================================
# CLEANUP - Comportamento Pos-Processamento
# ============================================================================

# Apagar pasta de input apos sucesso?
DELETE_INPUT_AFTER_SUCCESS = False

# Apagar temp/ apos sucesso?
DELETE_TEMP_AFTER_SUCCESS = True

# Apagar checkpoints/ apos sucesso?
DELETE_CHECKPOINTS_AFTER_SUCCESS = True

# ============================================================================
# HARDWARE - GPU/CPU/RAM
# ============================================================================

# Usar GPU (fallback automatico para CPU se indisponivel)
USE_GPU = True

# Device especifico (None = auto-detect)
CUDA_DEVICE = None  # None, 0, 1, etc.

# Memoria GPU maxima (% de uso)
MAX_GPU_MEMORY_USE = 0.85  # 85% de 15GB = ~12.7GB

# Memoria RAM maxima (% de uso)
MAX_RAM_USE = 0.80  # 80% de 30GB = ~24GB

# Habilitar monitoramento de RAM
ENABLE_RAM_MONITOR = True

# CPU workers (deixa 2 cores livres)
NUM_WORKERS = 6

# Paralelizacao de downloads (False = prioriza estabilidade)
PARALLEL_DOWNLOAD = False

# ============================================================================
# ETAPA 01: Normalizacao Inicial
# ============================================================================

# Formato do audio de entrada
INPUT_AUDIO_FORMAT = "flac"  # flac, mp3, wav, m4a, ogg

# Formato do audio de saida (normalizado)
OUTPUT_AUDIO_FORMAT = "flac"

# Sample rate para processamento interno
PREPROCESSING_SAMPLE_RATE = 24000  # Hz

# Compressao FLAC (0-8, padrao: 5)
FLAC_COMPRESSION_LEVEL = 5

# Timeout FFmpeg (segundos)
FFMPEG_TIMEOUT = 300

# Canais de saida (1=mono, 2=stereo)
OUTPUT_CHANNELS = 1

# ============================================================================
# ETAPA 02: Segmentacao Inteligente
# ============================================================================

# Validacao de audio ORIGINAL (antes de segmentar)
AUDIO_MIN_DURATION_TOLERANCE = 10     # segundos (rejeita se menor)
AUDIO_MAX_DURATION_TOLERANCE = 7200   # segundos (2h, rejeita se maior)

# Duracao dos segmentos gerados
SEGMENT_MIN_DURATION = 4.0   # segundos
SEGMENT_MAX_DURATION = 18.0  # segundos

# Overlap entre segmentos (evita perder palavras nas bordas)
SEGMENT_OVERLAP = 0.5  # segundos

# WebRTC VAD - Configuracoes
VAD_MODE = 2  # 0=menos agressivo, 3=mais agressivo (2=balanceado)
VAD_FRAME_DURATION = 30  # ms (10, 20 ou 30)

# Deteccao de silencio
SILENCE_THRESHOLD_DB = -35  # dB (pausas naturais)
MIN_SILENCE_DURATION = 0.2  # segundos (minimo para cortar)
MAX_SILENCE_DURATION = 2.0  # segundos (maximo pausa natural)

# Analise espectral
ENERGY_THRESHOLD = 0.01  # threshold minimo de energia
SPECTRAL_CENTROID_THRESHOLD = 1000  # Hz (deteccao de fala)

# ============================================================================
# ETAPA 03: Filtro MOS
# ============================================================================

# Threshold MOS (3-tier)
MOS_THRESHOLD_LOW = 2.5      # Abaixo = rejeitado
MOS_THRESHOLD_HIGH = 3.0     # Acima = aprovado direto

# Batch size para MOS
BATCH_SIZE_MOS = 15

# ============================================================================
# ETAPA 04-06: Diarizacao
# ============================================================================

# HuggingFace token (obrigatorio para pyannote)
HUGGINGFACE_TOKEN = ""  # Preencher

# Modelo de diarizacao
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

# ============================================================================
# ETAPA 08-09: STT
# ============================================================================

# Modelos STT
WHISPER_MODEL = "freds0/distil-whisper-large-v3-ptbr"
WAV2VEC2_MODEL = "lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2"

# Batch size para STT
BATCH_SIZE_STT = 10

# ============================================================================
# ETAPA 11: Validacao
# ============================================================================

# Threshold Levenshtein (0.0-1.0)
LEVENSHTEIN_THRESHOLD = 0.80

# ============================================================================
# ETAPA 13: Sox Final
# ============================================================================

# Sample rate final
FINAL_SAMPLE_RATE = 48000  # Hz

# Normalizacao de gain
NORMALIZE_GAIN = True

# ============================================================================
# LOGGING
# ============================================================================

# Nivel de log (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = "INFO"

# Habilitar progress bars (tqdm)
ENABLE_PROGRESS_BARS = True

# ============================================================================
# SISTEMA
# ============================================================================

# Modo dry-run (simula sem processar)
DRY_RUN = False

# Chunk size para I/O (bytes)
CHUNK_SIZE_IO = 1024 * 1024  # 1MB

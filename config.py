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
# ETAPA 03: Filtro MOS (SHEET 3-tier)
# ============================================================================

# Batch size para processamento MOS
BATCH_SIZE_MOS = 15  # ajustar conforme VRAM disponivel

# Thresholds MOS (escala 1.0-5.0)
MOS_THRESHOLD_LOW = 2.5   # Abaixo = rejeitado
MOS_THRESHOLD_HIGH = 3.0  # Acima ou igual = aprovado direto
# Entre 2.5-3.0 = intermediario (vai para denoising na etapa 12)

# Fallback automatico GPU->CPU se OOM
MOS_ENABLE_CPU_FALLBACK = True

# ============================================================================
# ETAPA 04: Diarizacao de Speakers
# ============================================================================

# Modelo pyannote (requer HuggingFace token em .env)
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Deteccao automatica de speakers (None = auto-detect)
EXPECTED_NUM_SPEAKERS = None  # int ou None

# Merge de segmentos consecutivos do mesmo speaker
# Gap maximo entre segmentos para mesclar (segundos)
DIARIZATION_MERGE_GAP = 0.5  # Se speaker fala novamente em <=0.5s, mescla

# Filtro de speakers por duracao minima
# Ignora speakers com menos de X segundos de fala total
MIN_SPEAKER_DURATION = 2.0  # segundos

# Duracao minima de um segmento individual de fala
# Remove segmentos muito curtos (provavelmente ruido)
MIN_SEGMENT_DURATION = 0.5  # segundos

# ============================================================================
# ETAPA 05: Overlap Detection (OSD)
# ============================================================================

# Modelo pyannote para deteccao de overlap
OVERLAP_DETECTION_MODEL = "pyannote/segmentation-3.0"

# Threshold de overlap para rejeitar segmento
# Se overlap >= threshold, o segmento eh descartado
# 0.9 = 90% do segmento tem overlap
OVERLAP_THRESHOLD = 0.9  # 0.0-1.0

# Duracao minima de overlap para considerar
# Ignora overlaps muito curtos (< X segundos)
MIN_OVERLAP_DURATION = 0.3  # segundos

# Analise de overlap em janelas
# Divide segmento em janelas para analise mais precisa
OVERLAP_WINDOW_SIZE = 1.0  # segundos
OVERLAP_HOP_SIZE = 0.5     # segundos (50% overlap)

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

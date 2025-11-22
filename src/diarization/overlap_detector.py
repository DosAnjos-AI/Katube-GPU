"""
Deteccao de sobreposicao de vozes (Overlap Speech Detection).
Identifica e rejeita segmentos com multiplos speakers falando simultaneamente.
"""
from pathlib import Path
from typing import List, Dict
import logging
import os
import time
import shutil

import numpy as np
import soundfile as sf

from src.core.timer import Timer

logger = logging.getLogger(__name__)

# Imports opcionais para pyannote
try:
    import torch
    from pyannote.audio import Model
    HAS_PYANNOTE = True
except ImportError:
    HAS_PYANNOTE = False


class OverlapDetector:
    """
    Detecta overlap de vozes em segmentos de audio.

    Usa pyannote/segmentation-3.0 para identificar regioes onde
    2+ speakers falam simultaneamente. Segmentos com overlap >= threshold
    sao descartados para garantir qualidade TTS.
    """

    # Cache singleton do modelo (carrega 1x, reutiliza)
    _model_cache = None

    def __init__(self, config):
        """
        Inicializa detector de overlap.

        Args:
            config: Modulo de configuracao
        """
        if not HAS_PYANNOTE:
            raise ImportError(
                "pyannote.audio nao instalado. "
                "Instale com: pip install pyannote.audio"
            )

        self.model_name = config.OVERLAP_DETECTION_MODEL
        self.overlap_threshold = config.OVERLAP_THRESHOLD
        self.min_overlap_duration = config.MIN_OVERLAP_DURATION
        self.window_size = config.OVERLAP_WINDOW_SIZE
        self.hop_size = config.OVERLAP_HOP_SIZE

        # Carregar token HuggingFace
        self.hf_token = self._load_hf_token()

        # Detectar device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Carregar modelo (cache singleton)
        self.model = self._load_model()

        logger.info(
            f"OverlapDetector inicializado "
            f"(device={self.device}, threshold={self.overlap_threshold})"
        )

    def _load_hf_token(self) -> str:
        """
        Carrega token HuggingFace do ambiente ou .env.

        Returns:
            Token HuggingFace

        Raises:
            RuntimeError: Se token nao encontrado
        """
        # Tentar carregar do .env
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        token = os.getenv('HUGGINGFACE_TOKEN')

        if not token:
            raise RuntimeError(
                "HUGGINGFACE_TOKEN nao encontrado no .env\n"
                "Configure em .env: HUGGINGFACE_TOKEN=hf_xxxxx"
            )

        logger.debug("Token HuggingFace carregado")
        return token

    def _load_model(self):
        """
        Carrega modelo pyannote segmentation (cache singleton).

        Returns:
            Modelo pyannote carregado
        """
        if OverlapDetector._model_cache is None:
            logger.info("Carregando modelo pyannote segmentation (primeira vez)...")
            try:
                # Carregar modelo
                OverlapDetector._model_cache = Model.from_pretrained(
                    self.model_name,
                    use_auth_token=self.hf_token
                )

                # Mover para GPU se disponivel
                OverlapDetector._model_cache.to(self.device)

                logger.info("Modelo pyannote segmentation carregado com sucesso")
            except Exception as e:
                logger.error(f"Erro ao carregar modelo segmentation: {e}")
                raise RuntimeError(
                    f"Falha ao carregar modelo de overlap detection: {e}\n"
                    "Verifique:\n"
                    "1. Token HuggingFace valido em .env\n"
                    "2. Termos aceitos em https://huggingface.co/pyannote/segmentation-3.0"
                )
        else:
            logger.debug("Reutilizando modelo segmentation do cache")

        return OverlapDetector._model_cache

    def detect_overlap_in_segment(
        self,
        audio_path: Path,
        segment_id: str
    ) -> Dict:
        """
        Detecta overlap em um segmento de audio.

        Args:
            audio_path: Caminho do arquivo de audio
            segment_id: ID do segmento (ex: 'seg_001')

        Returns:
            Dict com resultados da deteccao de overlap
        """
        start_time = time.time()

        try:
            # Carregar audio
            audio, sr = sf.read(audio_path)

            # Converter para mono se stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)

            audio_duration = len(audio) / sr

            # Converter para tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

            # Inferencia do modelo
            with torch.no_grad():
                output = self.model(audio_tensor)

            # Output shape: (batch, frames, classes)
            # Classes: [speech, overlap] ou similar
            # Pegamos a ultima classe como overlap
            overlap_probs = output[0, :, -1].cpu().numpy()

            # Converter frames para tempo
            frame_duration = audio_duration / len(overlap_probs)

            # Detectar regioes de overlap (threshold 0.5 na probabilidade)
            overlap_mask = overlap_probs > 0.5
            overlap_regions = self._extract_overlap_regions(
                overlap_mask,
                frame_duration
            )

            # Filtrar overlaps muito curtos
            overlap_regions = [
                region for region in overlap_regions
                if region['duration'] >= self.min_overlap_duration
            ]

            # Calcular metricas
            total_overlap_duration = sum(r['duration'] for r in overlap_regions)
            overlap_ratio = total_overlap_duration / audio_duration if audio_duration > 0 else 0.0
            has_overlap = overlap_ratio > 0.0
            rejected = overlap_ratio >= self.overlap_threshold

            processing_time = time.time() - start_time

            result = {
                'success': True,
                'segment_id': segment_id,
                'audio_path': str(audio_path),
                'has_overlap': has_overlap,
                'overlap_ratio': round(overlap_ratio, 4),
                'total_overlap_duration': round(total_overlap_duration, 2),
                'audio_duration': round(audio_duration, 2),
                'overlap_regions': overlap_regions,
                'rejected': rejected,
                'processing_time_s': round(processing_time, 2)
            }

            # Log resultado
            status = "REJEITADO" if rejected else "APROVADO"
            logger.info(
                f"{segment_id}: {status} "
                f"(overlap={overlap_ratio:.1%}, duration={audio_duration:.1f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"Erro ao detectar overlap em {segment_id}: {e}")
            return {
                'success': False,
                'segment_id': segment_id,
                'audio_path': str(audio_path),
                'error': str(e),
                'rejected': True  # Rejeitar em caso de erro
            }

    def _extract_overlap_regions(
        self,
        overlap_mask: np.ndarray,
        frame_duration: float
    ) -> List[Dict]:
        """
        Extrai regioes continuas de overlap.

        Args:
            overlap_mask: Array booleano (True = overlap)
            frame_duration: Duracao de cada frame em segundos

        Returns:
            Lista de regioes {start, end, duration}
        """
        regions = []
        in_overlap = False
        start_frame = 0

        for i, is_overlap in enumerate(overlap_mask):
            if is_overlap and not in_overlap:
                # Inicio de regiao de overlap
                start_frame = i
                in_overlap = True
            elif not is_overlap and in_overlap:
                # Fim de regiao de overlap
                end_frame = i
                start_time = start_frame * frame_duration
                end_time = end_frame * frame_duration
                duration = end_time - start_time

                regions.append({
                    'start': round(start_time, 2),
                    'end': round(end_time, 2),
                    'duration': round(duration, 2)
                })

                in_overlap = False

        # Ultima regiao se terminou em overlap
        if in_overlap:
            end_frame = len(overlap_mask)
            start_time = start_frame * frame_duration
            end_time = end_frame * frame_duration
            duration = end_time - start_time

            regions.append({
                'start': round(start_time, 2),
                'end': round(end_time, 2),
                'duration': round(duration, 2)
            })

        return regions

    def process_batch(
        self,
        segments: List[Dict],
        output_base_dir: Path
    ) -> Dict:
        """
        Processa batch de segmentos para deteccao de overlap.

        Args:
            segments: Lista de dicts com {'segment_id', 'audio_path', 'absolute_start'}
            output_base_dir: Diretorio base para outputs

        Returns:
            Dict com resultados do processamento em batch
        """
        logger.info(f"Iniciando deteccao de overlap em {len(segments)} segmentos")

        # Criar pasta para rejeitados
        rejected_dir = output_base_dir / 'rejeitados_overlap'
        rejected_dir.mkdir(parents=True, exist_ok=True)

        approved = []
        rejected = []
        failed = []
        overlap_ratios = []
        total_overlap_duration = 0.0

        for i, segment in enumerate(segments, 1):
            segment_id = segment['segment_id']
            audio_path = Path(segment.get('audio_path', segment.get('file_path', '')))

            logger.debug(f"[{i}/{len(segments)}] Analisando {segment_id}")

            # Detectar overlap
            result = self.detect_overlap_in_segment(audio_path, segment_id)

            if not result['success']:
                # Erro na deteccao
                failed.append({**segment, 'error': result.get('error')})
                continue

            # Atualizar segmento com resultado
            segment_with_result = {
                **segment,
                'overlap_detection': {
                    'has_overlap': result['has_overlap'],
                    'overlap_ratio': result['overlap_ratio'],
                    'total_overlap_duration': result['total_overlap_duration'],
                    'overlap_regions': result['overlap_regions']
                }
            }

            if result['rejected']:
                # Rejeitado por overlap
                rejected.append(segment_with_result)

                # Mover audio para pasta de rejeitados
                if audio_path.exists():
                    rejected_path = rejected_dir / audio_path.name
                    shutil.move(str(audio_path), str(rejected_path))
                    logger.debug(f"Movido para rejeitados: {audio_path.name}")

            else:
                # Aprovado
                approved.append(segment_with_result)
                overlap_ratios.append(result['overlap_ratio'])
                total_overlap_duration += result['total_overlap_duration']

        # Calcular estatisticas
        stats = {
            'avg_overlap_ratio': round(float(np.mean(overlap_ratios)), 4) if overlap_ratios else 0.0,
            'max_overlap_ratio': round(float(np.max(overlap_ratios)), 4) if overlap_ratios else 0.0,
            'total_overlap_duration': round(total_overlap_duration, 2)
        }

        logger.info(
            f"Overlap detection concluida: "
            f"{len(approved)} aprovados, {len(rejected)} rejeitados, "
            f"{len(failed)} falhas"
        )

        return {
            'success': True,
            'total_segments': len(segments),
            'approved_count': len(approved),
            'rejected_count': len(rejected),
            'failed_count': len(failed),
            'approved': approved,
            'rejected': rejected,
            'failed': failed,
            'stats': stats
        }

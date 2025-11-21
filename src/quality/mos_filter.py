"""
Filtro de qualidade de audio usando MOS (Mean Opinion Score).
Modelo: SHEET (Speech-to-Hearing Evaluation with Enhanced Terminology)
"""
from pathlib import Path
from typing import List, Dict
import logging
import shutil

import torch

from src.core.timer import Timer

logger = logging.getLogger(__name__)


class MOSFilter:
    """
    Filtro de qualidade baseado em MOS score.

    Classifica audios em 3 tiers:
    - Rejeitado: MOS < 2.5
    - Intermediario: 2.5 <= MOS < 3.0 (vai para denoising)
    - Aprovado: MOS >= 3.0
    """

    # Cache singleton do modelo (carrega 1x, reutiliza)
    _model_cache = None
    _device_cache = None

    def __init__(self, config):
        """
        Inicializa filtro MOS.

        Args:
            config: Modulo de configuracao
        """
        self.batch_size = config.BATCH_SIZE_MOS
        self.threshold_low = config.MOS_THRESHOLD_LOW
        self.threshold_high = config.MOS_THRESHOLD_HIGH
        self.enable_cpu_fallback = config.MOS_ENABLE_CPU_FALLBACK

        # Detectar device
        self.device = self._get_device(config.USE_GPU)

        # Carregar modelo (cache singleton)
        self.model = self._load_model()

        logger.info(
            f"MOSFilter inicializado "
            f"(device={self.device}, batch={self.batch_size})"
        )

    def _get_device(self, use_gpu: bool) -> str:
        """
        Detecta melhor device disponivel.

        Args:
            use_gpu: Se deve tentar usar GPU

        Returns:
            'cuda' ou 'cpu'
        """
        if use_gpu and torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detectada: {gpu_name}")
        else:
            device = 'cpu'
            if use_gpu:
                logger.warning("GPU nao disponivel, usando CPU")
            else:
                logger.info("Usando CPU (conforme config)")

        return device

    @classmethod
    def _load_model(cls):
        """
        Carrega modelo SHEET (cache singleton).

        Returns:
            Modelo SHEET carregado
        """
        if cls._model_cache is None:
            logger.info("Carregando modelo SHEET (primeira vez)...")
            try:
                # Carregar via torch.hub
                cls._model_cache = torch.hub.load(
                    "unilight/sheet:v0.1.0",
                    "default",
                    trust_repo=True,
                    force_reload=False
                )
                logger.info("Modelo SHEET carregado com sucesso")
            except Exception as e:
                logger.error(f"Erro ao carregar modelo SHEET: {e}")
                raise RuntimeError(
                    f"Falha ao carregar modelo MOS: {e}\n"
                    "Verifique conexao com internet e torch.hub"
                )
        else:
            logger.debug("Reutilizando modelo SHEET do cache")

        return cls._model_cache

    def predict_mos(self, audio_path: Path) -> float:
        """
        Prediz score MOS para um audio.

        Args:
            audio_path: Caminho do arquivo de audio

        Returns:
            Score MOS (1.0-5.0)

        Raises:
            RuntimeError: Se predicao falhar em ambos GPU e CPU
        """
        try:
            # Tentar GPU primeiro
            if self.device == 'cuda':
                try:
                    self.model.model.cuda()
                    score = self.model.predict(wav_path=str(audio_path))
                    return float(score)
                except RuntimeError as e:
                    # OOM ou erro GPU
                    if self.enable_cpu_fallback:
                        logger.warning(
                            f"Erro GPU ({e}), tentando CPU fallback..."
                        )
                        self.model.model.cpu()
                        score = self.model.predict(wav_path=str(audio_path))
                        return float(score)
                    else:
                        raise
            else:
                # CPU direto
                self.model.model.cpu()
                score = self.model.predict(wav_path=str(audio_path))
                return float(score)

        except Exception as e:
            logger.error(f"Erro ao predizer MOS para {audio_path.name}: {e}")
            raise RuntimeError(f"Predicao MOS falhou: {e}")

    def classify_tier(self, mos_score: float) -> str:
        """
        Classifica tier baseado no score MOS.

        Args:
            mos_score: Score MOS (1.0-5.0)

        Returns:
            'rejected', 'intermediate' ou 'approved'
        """
        if mos_score < self.threshold_low:
            return 'rejected'
        elif mos_score < self.threshold_high:
            return 'intermediate'
        else:
            return 'approved'

    def filter_segments(
        self,
        segments: List[Dict],
        output_base_dir: Path
    ) -> Dict:
        """
        Filtra segmentos por qualidade MOS.

        Args:
            segments: Lista de segmentos do PROMPT 02
                [{'file_path': Path, 'segment_id': str, ...}, ...]
            output_base_dir: Diretorio base para salvar resultados

        Returns:
            Dict com:
            {
                'success': bool,
                'approved': [{'segment_id': str, 'mos_score': float, ...}],
                'intermediate': [...],
                'rejected': [...],
                'total_segments': int,
                'processing_time_s': float,
                'stats': {
                    'approved_count': int,
                    'intermediate_count': int,
                    'rejected_count': int,
                    'min_score': float,
                    'max_score': float,
                    'avg_score': float
                },
                'error': str (se falhar)
            }
        """
        with Timer("Filtro MOS") as timer:
            try:
                # Criar pastas de output
                approved_dir = output_base_dir / 'mos_results' / 'approved'
                intermediate_dir = output_base_dir / 'mos_results' / 'intermediate'
                rejected_dir = output_base_dir / 'mos_results' / 'rejected'

                approved_dir.mkdir(parents=True, exist_ok=True)
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                rejected_dir.mkdir(parents=True, exist_ok=True)

                # Resultados
                approved = []
                intermediate = []
                rejected = []
                all_scores = []

                total = len(segments)
                logger.info(f"Processando {total} segmentos...")

                # Processar segmentos
                for i, segment in enumerate(segments):
                    segment_id = segment['segment_id']
                    file_path = segment['file_path']

                    # Converter para Path se for string
                    if isinstance(file_path, str):
                        file_path = Path(file_path)

                    try:
                        # Predizer MOS
                        mos_score = self.predict_mos(file_path)
                        all_scores.append(mos_score)

                        # Classificar tier
                        tier = self.classify_tier(mos_score)

                        # Preparar dados do segmento
                        segment_data = {
                            **segment,  # Preservar dados do PROMPT 02
                            'mos_score': mos_score,
                            'mos_tier': tier
                        }

                        # Copiar para pasta apropriada
                        if tier == 'approved':
                            dest_path = approved_dir / file_path.name
                            approved.append(segment_data)
                        elif tier == 'intermediate':
                            dest_path = intermediate_dir / file_path.name
                            intermediate.append(segment_data)
                        else:  # rejected
                            dest_path = rejected_dir / file_path.name
                            rejected.append(segment_data)

                        shutil.copy2(file_path, dest_path)

                        # Log incremental (a cada 10 segmentos)
                        if (i + 1) % 10 == 0 or (i + 1) == total:
                            logger.info(f"Processados: {i+1}/{total} segmentos")

                    except RuntimeError as e:
                        # Erro MOS: REJEITAR segmento
                        logger.error(
                            f"MOS falhou para {segment_id}, rejeitando: {e}"
                        )
                        segment_data = {
                            **segment,
                            'mos_score': 0.0,
                            'mos_tier': 'rejected',
                            'mos_error': str(e)
                        }
                        rejected.append(segment_data)

                # Calcular estatisticas
                stats = {
                    'approved_count': len(approved),
                    'intermediate_count': len(intermediate),
                    'rejected_count': len(rejected),
                    'min_score': min(all_scores) if all_scores else 0.0,
                    'max_score': max(all_scores) if all_scores else 0.0,
                    'avg_score': (
                        sum(all_scores) / len(all_scores)
                        if all_scores else 0.0
                    )
                }

                logger.info(
                    f"MOS completo: {stats['approved_count']} aprovados, "
                    f"{stats['intermediate_count']} intermediarios, "
                    f"{stats['rejected_count']} rejeitados "
                    f"(media: {stats['avg_score']:.2f})"
                )

                return {
                    'success': True,
                    'approved': approved,
                    'intermediate': intermediate,
                    'rejected': rejected,
                    'total_segments': total,
                    'processing_time_s': timer.elapsed_time,
                    'stats': stats
                }

            except Exception as e:
                logger.error(f"Erro no filtro MOS: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'processing_time_s': timer.elapsed_time if timer.elapsed_time else 0.0
                }

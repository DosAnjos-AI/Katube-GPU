"""
Gerenciador de checkpoints para resume da pipeline.
Salva estado apos cada etapa concluida.
"""
from pathlib import Path
from typing import Dict, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Gerencia checkpoints da pipeline.

    Salva em: dataset/checkpoints/{video_id}.json
    """

    def __init__(self, config):
        """
        Inicializa gerenciador.

        Args:
            config: Modulo de configuracao
        """
        self.checkpoint_dir = Path("dataset/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        video_id: str,
        etapa: str,
        data: Dict
    ):
        """
        Salva checkpoint de uma etapa.

        Args:
            video_id: ID do video
            etapa: Nome da etapa (ex: "02_segmentacao")
            data: Dados da etapa para salvar
        """
        checkpoint_path = self.checkpoint_dir / f"{video_id}.json"

        # Carregar checkpoint existente (se houver)
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
        else:
            checkpoint = {
                'video_id': video_id,
                'etapas': {}
            }

        # Converter Path objects para strings no data
        data_serializable = self._make_serializable(data)

        # Adicionar/atualizar etapa
        checkpoint['etapas'][etapa] = {
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'data': data_serializable
        }

        # Atualizar ultima etapa concluida
        checkpoint['ultima_etapa'] = etapa

        # Salvar
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        logger.info(f"Checkpoint salvo: {etapa}")

    def _make_serializable(self, obj):
        """
        Converte objetos nao serializaveis para JSON.

        Args:
            obj: Objeto a converter

        Returns:
            Objeto serializavel
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    def load_checkpoint(self, video_id: str) -> Optional[Dict]:
        """
        Carrega checkpoint de um video.

        Args:
            video_id: ID do video

        Returns:
            Dict com checkpoint ou None se nao existir
        """
        checkpoint_path = self.checkpoint_dir / f"{video_id}.json"

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar checkpoint: {e}")
            return None

    def get_last_completed_stage(self, video_id: str) -> Optional[str]:
        """
        Retorna ultima etapa concluida de um video.

        Args:
            video_id: ID do video

        Returns:
            Nome da etapa ou None
        """
        checkpoint = self.load_checkpoint(video_id)
        if checkpoint:
            return checkpoint.get('ultima_etapa')
        return None

    def delete_checkpoint(self, video_id: str) -> bool:
        """
        Remove checkpoint de um video.

        Args:
            video_id: ID do video

        Returns:
            True se removido, False se nao existia
        """
        checkpoint_path = self.checkpoint_dir / f"{video_id}.json"

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Checkpoint removido: {video_id}")
            return True

        return False

    def update_segments_with_mos(
        self,
        video_id: str,
        mos_results: Dict
    ):
        """
        Atualiza checkpoint adicionando scores MOS aos segmentos existentes.

        Args:
            video_id: ID do video
            mos_results: Resultado do filtro MOS
        """
        checkpoint = self.load_checkpoint(video_id)

        if not checkpoint:
            logger.error(f"Checkpoint nao encontrado para {video_id}")
            return

        # Obter segmentos do checkpoint da segmentacao
        if '02_segmentacao' not in checkpoint['etapas']:
            logger.error("Checkpoint de segmentacao nao encontrado")
            return

        segments_data = checkpoint['etapas']['02_segmentacao']['data']['segments']

        # Merge MOS scores
        # Combinar approved, intermediate, rejected
        all_mos = (
            mos_results.get('approved', []) +
            mos_results.get('intermediate', []) +
            mos_results.get('rejected', [])
        )

        for mos_segment in all_mos:
            segment_id = mos_segment['segment_id']

            # Atualizar segmento existente
            for seg in segments_data:
                if seg['segment_id'] == segment_id:
                    seg['mos_score'] = mos_segment['mos_score']
                    seg['mos_tier'] = mos_segment['mos_tier']
                    if 'mos_error' in mos_segment:
                        seg['mos_error'] = mos_segment['mos_error']
                    break

        # Salvar checkpoint atualizado
        checkpoint['etapas']['02_segmentacao']['data']['segments'] = segments_data

        # Salvar diretamente no arquivo
        checkpoint_path = self.checkpoint_dir / f"{video_id}.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        # Adicionar etapa MOS
        self.save_checkpoint(
            video_id=video_id,
            etapa='03_mos_filter',
            data={
                'processing_time_s': mos_results.get('processing_time_s', 0.0),
                'stats': mos_results.get('stats', {})
            }
        )

        logger.info(f"Checkpoint atualizado com scores MOS")

    def save_diarization_results(
        self,
        video_id: str,
        diarization_results: Dict
    ) -> None:
        """
        Salva resultados da diarizacao no checkpoint.

        Args:
            video_id: ID do video
            diarization_results: Resultado do Diarizer.diarize_batch()
        """
        # Estruturar dados para checkpoint
        checkpoint_data = {
            'total_segments': diarization_results['total_segments'],
            'processed_count': len(diarization_results['processed']),
            'failed_count': len(diarization_results['failed']),
            'stats': diarization_results['stats'],
            'segments': {}
        }

        # Adicionar dados de cada segmento processado
        for result in diarization_results['processed']:
            segment_id = result['segment_id']
            checkpoint_data['segments'][segment_id] = {
                'num_speakers': result['num_speakers'],
                'speakers': result['speakers'],
                'rttm_path': result['rttm_path'],
                'json_path': result['json_path']
            }

        # Salvar no checkpoint
        self.save_checkpoint(
            video_id=video_id,
            etapa='04_diarizacao',
            data=checkpoint_data
        )

        logger.info(f"Checkpoint atualizado com dados de diarizacao")

    def save_overlap_detection_results(
        self,
        video_id: str,
        results: Dict
    ) -> None:
        """
        Salva resultados da deteccao de overlap no checkpoint.

        Args:
            video_id: ID do video
            results: Resultados do OverlapDetector.process_batch()
        """
        # Extrair dados principais
        data = {
            'total_segments': results['total_segments'],
            'approved_count': results['approved_count'],
            'rejected_count': results['rejected_count'],
            'failed_count': results['failed_count'],
            'stats': results['stats'],
            'approved': self._make_serializable(results['approved']),
            'rejected': [
                {
                    'segment_id': seg['segment_id'],
                    'audio_path': seg.get('audio_path', seg.get('file_path', '')),
                    'overlap_ratio': seg.get('overlap_detection', {}).get('overlap_ratio', 0.0)
                }
                for seg in results['rejected']
            ]
        }

        # Salvar no checkpoint
        self.save_checkpoint(
            video_id=video_id,
            etapa='05_overlap_detection',
            data=data
        )

        logger.info(
            f"Checkpoint atualizado (overlap detection): "
            f"{results['approved_count']} aprovados, "
            f"{results['rejected_count']} rejeitados"
        )

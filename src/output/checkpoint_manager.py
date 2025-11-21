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

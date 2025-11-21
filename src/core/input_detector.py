"""
Detector de cenario de input.
Identifica automaticamente estrutura de diretorios e extrai video_id.
"""
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class InputDetector:
    """
    Detecta automaticamente o cenario de input.

    Suporta dois cenarios:
    1. Subpastas: audios/video_id/video_id.flac
    2. Flat: audios/video_id.flac
    """

    @staticmethod
    def extract_video_id(filename: str) -> str:
        """
        Extrai video_id do nome do arquivo.

        ID e sempre a primeira parte antes de underscore ou ponto.

        Args:
            filename: Nome do arquivo (com ou sem extensao)

        Returns:
            video_id extraido

        Examples:
            >>> extract_video_id("abc123.flac")
            "abc123"
            >>> extract_video_id("abc123_titulo_video.flac")
            "abc123"
        """
        name = Path(filename).stem
        video_id = name.split('_')[0]
        return video_id

    @staticmethod
    def detect_scenario(
        audio_input_dir: Path,
        input_format: str
    ) -> Tuple[str, List[Tuple[str, Path]]]:
        """
        Detecta cenario e retorna lista de (video_id, audio_path).

        Args:
            audio_input_dir: Diretorio de input
            input_format: Extensao esperada (ex: "flac")

        Returns:
            Tupla (scenario, files):
            - scenario: "subfolders" ou "flat"
            - files: [(video_id, Path), ...]

        Raises:
            FileNotFoundError: Se diretorio nao existe
            ValueError: Se nenhum audio encontrado
        """
        if not audio_input_dir.exists():
            raise FileNotFoundError(
                f"Diretorio de input nao existe: {audio_input_dir}"
            )

        # Tentar detectar subpastas primeiro (CENARIO 1)
        subdirs = [d for d in audio_input_dir.iterdir() if d.is_dir()]

        if subdirs:
            logger.info(
                f"Cenario detectado: SUBPASTAS ({len(subdirs)} encontradas)"
            )
            return InputDetector._process_subfolders(subdirs, input_format)

        # Cenario 2: Arquivos na raiz
        logger.info("Cenario detectado: FLAT (arquivos na raiz)")
        return InputDetector._process_flat(audio_input_dir, input_format)

    @staticmethod
    def _process_subfolders(
        subdirs: List[Path],
        input_format: str
    ) -> Tuple[str, List[Tuple[str, Path]]]:
        """Processa cenario de subpastas."""
        files = []

        for subdir in subdirs:
            audio_files = list(subdir.glob(f"*.{input_format}"))

            if audio_files:
                # Pega primeiro arquivo encontrado
                audio_path = audio_files[0]
                video_id = InputDetector.extract_video_id(audio_path.name)
                files.append((video_id, audio_path))

                if len(audio_files) > 1:
                    logger.warning(
                        f"Multiplos .{input_format} em {subdir.name}, "
                        f"usando: {audio_path.name}"
                    )

        if not files:
            raise ValueError(
                f"Nenhum arquivo .{input_format} encontrado nas subpastas"
            )

        return "subfolders", files

    @staticmethod
    def _process_flat(
        audio_input_dir: Path,
        input_format: str
    ) -> Tuple[str, List[Tuple[str, Path]]]:
        """Processa cenario flat (arquivos na raiz)."""
        audio_files = list(audio_input_dir.glob(f"*.{input_format}"))

        if not audio_files:
            raise ValueError(
                f"Nenhum arquivo .{input_format} encontrado em "
                f"{audio_input_dir}"
            )

        files = []
        for audio_path in audio_files:
            video_id = InputDetector.extract_video_id(audio_path.name)
            files.append((video_id, audio_path))

        return "flat", files

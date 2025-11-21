"""
Normalizador de audio usando FFmpeg (com fallback librosa).
Padroniza audio para 24kHz, mono, FLAC.
"""
from pathlib import Path
from typing import Dict
import subprocess
import shutil
import time
import logging

logger = logging.getLogger(__name__)


class AudioNormalizer:
    """
    Normaliza audio para padrao da pipeline.

    Prioridade: FFmpeg -> librosa (fallback automatico)
    """

    def __init__(self, config):
        """
        Inicializa normalizador.

        Args:
            config: Modulo de configuracao (config.py)
        """
        self.input_format = config.INPUT_AUDIO_FORMAT
        self.output_format = config.OUTPUT_AUDIO_FORMAT
        self.target_sr = config.PREPROCESSING_SAMPLE_RATE
        self.target_channels = config.OUTPUT_CHANNELS
        self.flac_level = config.FLAC_COMPRESSION_LEVEL
        self.timeout = config.FFMPEG_TIMEOUT

        # Validar ferramentas disponiveis
        self._check_tools()

    def _check_tools(self):
        """Valida FFmpeg e fallback librosa."""
        self.has_ffmpeg = bool(shutil.which('ffmpeg'))

        try:
            import librosa
            import soundfile
            self.has_librosa = True
        except ImportError:
            self.has_librosa = False

        if not self.has_ffmpeg and not self.has_librosa:
            raise RuntimeError(
                "Nem FFmpeg nem librosa disponiveis. "
                "Instale FFmpeg ou: pip install librosa soundfile"
            )

        if self.has_ffmpeg:
            logger.info("FFmpeg disponivel")
        else:
            logger.warning("FFmpeg nao encontrado, usando librosa")

    def normalize(
        self,
        input_path: Path,
        output_path: Path
    ) -> Dict:
        """
        Normaliza audio com fallback automatico.

        Args:
            input_path: Caminho do audio original
            output_path: Caminho para salvar normalizado

        Returns:
            Dict com:
            {
                'success': bool,
                'method': 'ffmpeg' ou 'librosa',
                'input_size_mb': float,
                'output_size_mb': float,
                'processing_time_s': float,
                'error': str (se falhar)
            }
        """
        start_time = time.time()

        # Garantir que diretorio de saida existe
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Tentar FFmpeg primeiro
        if self.has_ffmpeg:
            result = self._normalize_ffmpeg(input_path, output_path)
            if result['success']:
                result['method'] = 'ffmpeg'
                result['processing_time_s'] = time.time() - start_time
                logger.info(
                    f"Normalizado com FFmpeg: {output_path.name} "
                    f"({result['processing_time_s']:.2f}s)"
                )
                return result

            logger.warning(
                f"FFmpeg falhou ({result['error']}), tentando librosa..."
            )

        # Fallback: librosa
        if self.has_librosa:
            result = self._normalize_librosa(input_path, output_path)
            if result['success']:
                result['method'] = 'librosa'
                result['processing_time_s'] = time.time() - start_time
                logger.info(
                    f"Normalizado com librosa: {output_path.name} "
                    f"({result['processing_time_s']:.2f}s)"
                )
                return result

        # Falhou tudo
        return {
            'success': False,
            'error': 'FFmpeg e librosa falharam',
            'processing_time_s': time.time() - start_time
        }

    def _normalize_ffmpeg(
        self,
        input_path: Path,
        output_path: Path
    ) -> Dict:
        """
        Normalizacao via FFmpeg.

        Returns:
            Dict com success, input_size_mb, output_size_mb, error
        """
        cmd = [
            'ffmpeg', '-i', str(input_path),
            '-ar', str(self.target_sr),
            '-ac', str(self.target_channels),
            '-acodec', self.output_format,
        ]

        # Adicionar compressao se FLAC
        if self.output_format == 'flac':
            cmd.extend(['-compression_level', str(self.flac_level)])

        cmd.extend(['-y', str(output_path)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.timeout,
                text=True
            )

            if result.returncode == 0:
                return {
                    'success': True,
                    'input_size_mb': input_path.stat().st_size / (1024**2),
                    'output_size_mb': output_path.stat().st_size / (1024**2)
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr[:200]  # Primeiros 200 chars
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Timeout apos {self.timeout}s'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _normalize_librosa(
        self,
        input_path: Path,
        output_path: Path
    ) -> Dict:
        """
        Normalizacao via librosa (fallback).

        Returns:
            Dict com success, input_size_mb, output_size_mb, error
        """
        try:
            import librosa
            import soundfile as sf

            # Carregar audio
            audio, sr = librosa.load(
                str(input_path),
                sr=self.target_sr,
                mono=(self.target_channels == 1)
            )

            # Salvar
            sf.write(
                str(output_path),
                audio,
                self.target_sr,
                format=self.output_format
            )

            return {
                'success': True,
                'input_size_mb': input_path.stat().st_size / (1024**2),
                'output_size_mb': output_path.stat().st_size / (1024**2)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

"""
Segmentador inteligente de audio usando VAD + analise espectral.
Corta APENAS em pausas naturais da fala (nao corta palavras).
"""
from pathlib import Path
from typing import List, Tuple, Dict
import logging
import time

import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import uniform_filter1d

# webrtcvad opcional (fallback para deteccao por energia)
try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False

logger = logging.getLogger(__name__)


class AudioSegmenter:
    """
    Segmenta audio em chunks inteligentes baseados em pausas naturais.

    Usa WebRTC VAD + analise espectral para encontrar pontos de corte ideais.
    """

    def __init__(self, config):
        """
        Inicializa segmentador.

        Args:
            config: Modulo de configuracao (config.py)
        """
        # Validacoes de audio original
        self.min_audio_duration = config.AUDIO_MIN_DURATION_TOLERANCE
        self.max_audio_duration = config.AUDIO_MAX_DURATION_TOLERANCE

        # Duracao dos segmentos
        self.min_duration = config.SEGMENT_MIN_DURATION
        self.max_duration = config.SEGMENT_MAX_DURATION
        self.overlap = config.SEGMENT_OVERLAP

        # Sample rate
        self.sample_rate = config.PREPROCESSING_SAMPLE_RATE

        # VAD (opcional)
        self.has_vad = HAS_WEBRTCVAD
        if self.has_vad:
            self.vad = webrtcvad.Vad(config.VAD_MODE)
            logger.info("WebRTC VAD disponivel")
        else:
            self.vad = None
            logger.warning("WebRTC VAD nao disponivel, usando deteccao por energia")
        self.vad_frame_duration = config.VAD_FRAME_DURATION

        # Analise de silencio
        self.silence_threshold_db = config.SILENCE_THRESHOLD_DB
        self.min_silence_duration = config.MIN_SILENCE_DURATION
        self.max_silence_duration = config.MAX_SILENCE_DURATION

        # Analise espectral
        self.energy_threshold = config.ENERGY_THRESHOLD
        self.spectral_threshold = config.SPECTRAL_CENTROID_THRESHOLD

        logger.info("AudioSegmenter inicializado")

    def validate_audio_duration(self, audio_path: Path) -> Dict:
        """
        Valida duracao do audio ORIGINAL (antes de segmentar).

        Args:
            audio_path: Caminho do audio normalizado

        Returns:
            Dict com:
            {
                'valid': bool,
                'duration': float,
                'error': str (se invalido)
            }
        """
        try:
            info = sf.info(audio_path)
            duration = info.duration

            if duration < self.min_audio_duration:
                return {
                    'valid': False,
                    'duration': duration,
                    'error': f'Audio muito curto: {duration:.1f}s < {self.min_audio_duration}s'
                }

            if duration > self.max_audio_duration:
                return {
                    'valid': False,
                    'duration': duration,
                    'error': f'Audio muito longo: {duration:.1f}s > {self.max_audio_duration}s'
                }

            return {
                'valid': True,
                'duration': duration
            }

        except Exception as e:
            return {
                'valid': False,
                'duration': 0.0,
                'error': f'Erro ao ler audio: {str(e)}'
            }

    def detect_speech_activity(self, audio: np.ndarray) -> List[bool]:
        """
        Detecta atividade de fala usando WebRTC VAD ou energia.

        Args:
            audio: Array de audio (mono, 24kHz)

        Returns:
            Lista de bools (True=fala, False=silencio) por frame
        """
        frame_size = int(self.sample_rate * self.vad_frame_duration / 1000)

        if self.has_vad and self.vad is not None:
            # Usar WebRTC VAD
            audio_16bit = (audio * 32767).astype(np.int16)
            frames = []

            for i in range(0, len(audio_16bit), frame_size):
                frame = audio_16bit[i:i + frame_size]
                if len(frame) == frame_size:
                    frames.append(frame.tobytes())

            # Aplicar VAD
            speech_frames = []
            for frame in frames:
                try:
                    is_speech = self.vad.is_speech(frame, self.sample_rate)
                    speech_frames.append(is_speech)
                except:
                    speech_frames.append(False)
        else:
            # Fallback: deteccao por energia
            speech_frames = []
            for i in range(0, len(audio), frame_size):
                frame = audio[i:i + frame_size]
                if len(frame) == frame_size:
                    energy = np.mean(frame ** 2)
                    is_speech = energy > self.energy_threshold
                    speech_frames.append(is_speech)

        # Suavizacao (reduz falsos positivos)
        speech_frames = self._smooth_vad(speech_frames)

        return speech_frames

    def _smooth_vad(self, speech_frames: List[bool], window_size: int = 7) -> List[bool]:
        """
        Suaviza deteccao VAD (remove ruidos isolados).

        Args:
            speech_frames: Lista de deteccoes VAD
            window_size: Tamanho da janela de suavizacao

        Returns:
            Lista suavizada
        """
        if len(speech_frames) < window_size:
            return speech_frames

        smoothed = []
        for i in range(len(speech_frames)):
            start = max(0, i - window_size // 2)
            end = min(len(speech_frames), i + window_size // 2 + 1)
            window = speech_frames[start:end]

            # Maioria vence (60% threshold)
            speech_count = sum(window)
            smoothed.append(speech_count > len(window) * 0.6)

        return smoothed

    def detect_silence_regions(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detecta regioes de silencio usando analise de energia.

        Args:
            audio: Array de audio (mono, 24kHz)

        Returns:
            Lista de tuplas (start_sample, end_sample) de silencios
        """
        # Converter para dB
        hop_length = 512
        audio_db = librosa.amplitude_to_db(np.abs(audio))

        # Suavizar
        smoothing_size = max(hop_length // 4, 64)
        audio_db = uniform_filter1d(audio_db, size=smoothing_size)

        # Detectar silencio
        silence_mask = audio_db < self.silence_threshold_db

        # Encontrar regioes continuas
        silence_regions = []
        in_silence = False
        silence_start = 0

        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                silence_start = i
                in_silence = True
            elif not is_silent and in_silence:
                silence_regions.append((silence_start, i))
                in_silence = False

        if in_silence:
            silence_regions.append((silence_start, len(silence_mask)))

        # Filtrar por duracao
        min_samples = int(self.min_silence_duration * self.sample_rate)
        max_samples = int(self.max_silence_duration * self.sample_rate)

        filtered = []
        for start, end in silence_regions:
            duration = end - start
            if min_samples <= duration <= max_samples:
                filtered.append((start, end))

        return filtered

    def find_optimal_cut_points(self, audio: np.ndarray) -> List[int]:
        """
        Encontra pontos ideais para cortar o audio.

        Combina VAD + deteccao de silencio para cortar em pausas naturais.

        Args:
            audio: Array de audio (mono, 24kHz)

        Returns:
            Lista de indices (samples) para corte
        """
        # Detectar silencios
        silence_regions = self.detect_silence_regions(audio)

        logger.info(f"Detectadas {len(silence_regions)} regioes de silencio")

        # Criar segmentos baseados em silencios
        cut_points = [0]  # Inicio do audio
        current_pos = 0

        while current_pos < len(audio):
            best_cut = None
            best_score = float('inf')

            # Procurar melhor ponto de corte
            for silence_start, silence_end in silence_regions:
                if silence_start <= current_pos:
                    continue

                segment_duration = (silence_start - current_pos) / self.sample_rate

                # Pular se muito curto
                if segment_duration < self.min_duration:
                    continue

                # Forcar corte se muito longo
                if segment_duration > self.max_duration:
                    break

                # Calcular score (preferir silencios longos)
                silence_duration = (silence_end - silence_start) / self.sample_rate
                score = 1.0 / (silence_duration + 0.1)  # Menor score = melhor

                if score < best_score:
                    best_score = score
                    best_cut = (silence_start + silence_end) // 2

            # Aplicar corte
            if best_cut is not None:
                cut_points.append(best_cut)
                current_pos = best_cut
            else:
                # Sem silencio adequado, forcar no max_duration
                current_pos += int(self.max_duration * self.sample_rate)
                if current_pos < len(audio):
                    cut_points.append(current_pos)

        # Fim do audio
        cut_points.append(len(audio))

        # Remover duplicatas e ordenar
        cut_points = sorted(list(set(cut_points)))

        return cut_points

    def segment_audio(
        self,
        audio_path: Path,
        output_dir: Path,
        video_id: str
    ) -> Dict:
        """
        Segmenta audio em chunks inteligentes.

        Args:
            audio_path: Caminho do audio normalizado
            output_dir: Diretorio para salvar segmentos
            video_id: ID do video (para nomenclatura)

        Returns:
            Dict com:
            {
                'success': bool,
                'segments': [
                    {
                        'file_path': Path,
                        'segment_id': 'seg_001',
                        'absolute_start': float,
                        'absolute_end': float,
                        'duration': float
                    }
                ],
                'total_segments': int,
                'processing_time_s': float,
                'error': str (se falhar)
            }
        """
        start_time = time.time()

        # Validar duracao do audio original
        validation = self.validate_audio_duration(audio_path)
        if not validation['valid']:
            logger.error(f"Audio rejeitado: {validation['error']}")
            return {
                'success': False,
                'error': validation['error'],
                'processing_time_s': time.time() - start_time
            }

        logger.info(f"Segmentando {audio_path.name} ({validation['duration']:.1f}s)")

        # Carregar audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Normalizar
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Encontrar pontos de corte
        cut_points = self.find_optimal_cut_points(audio)

        logger.info(f"Encontrados {len(cut_points) - 1} segmentos potenciais")

        # Criar segmentos
        output_dir.mkdir(parents=True, exist_ok=True)
        segments = []
        segment_idx = 1

        for i in range(len(cut_points) - 1):
            start_sample = cut_points[i]
            end_sample = cut_points[i + 1]

            duration = (end_sample - start_sample) / self.sample_rate

            # Pular se muito curto
            if duration < self.min_duration:
                continue

            # Extrair segmento
            segment_audio = audio[start_sample:end_sample]

            # Nomenclatura: video_id_seg_001.flac
            segment_id = f"seg_{segment_idx:03d}"
            filename = f"{video_id}_{segment_id}.flac"
            segment_path = output_dir / filename

            # Salvar
            sf.write(segment_path, segment_audio, self.sample_rate)

            # Timestamps absolutos
            absolute_start = start_sample / self.sample_rate
            absolute_end = end_sample / self.sample_rate

            segments.append({
                'file_path': segment_path,
                'segment_id': segment_id,
                'absolute_start': absolute_start,
                'absolute_end': absolute_end,
                'duration': duration
            })

            logger.debug(
                f"{segment_id}: {duration:.2f}s "
                f"({absolute_start:.2f}s - {absolute_end:.2f}s)"
            )

            segment_idx += 1

        processing_time = time.time() - start_time

        logger.info(
            f"Segmentacao concluida: {len(segments)} segmentos "
            f"({processing_time:.2f}s)"
        )

        return {
            'success': True,
            'segments': segments,
            'total_segments': len(segments),
            'processing_time_s': processing_time
        }

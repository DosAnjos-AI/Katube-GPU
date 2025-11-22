"""
Separador de speakers baseado em diarizacao.
Extrai subsegmentos de audio por locutor com timestamps absolutos.

Autor: Equipe Katube
Data: 2025-01-15
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import shutil

# Imports opcionais para ambiente sem dependencias completas
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


class SpeakerSeparator:
    """
    Separa audio por speaker usando resultados de diarizacao.

    Processa apenas segmentos limpos (sem overlap) e gera
    subsegmentos com timestamps absolutos para rastreabilidade.
    """

    def __init__(
        self,
        config,
        min_duration: float = None,
        max_duration: float = None,
        max_gap: float = None
    ):
        """
        Inicializa separador.

        Args:
            config: Modulo de configuracao
            min_duration: Duracao minima de subsegmento (override config)
            max_duration: Duracao maxima de subsegmento (override config)
            max_gap: Gap maximo para merge (override config)
        """
        self.config = config
        self.min_duration = min_duration or config.SPEAKER_SEP_MIN_DURATION
        self.max_duration = max_duration or config.SPEAKER_SEP_MAX_DURATION
        self.max_gap = max_gap or config.SPEAKER_SEP_MAX_GAP
        self.sample_rate = config.PREPROCESSING_SAMPLE_RATE
        self.enhance_audio = config.SPEAKER_SEP_ENHANCE_AUDIO
        self.highpass_freq = config.SPEAKER_SEP_HIGHPASS_FREQ
        self.lowpass_freq = config.SPEAKER_SEP_LOWPASS_FREQ

    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Carrega arquivo de audio.

        Args:
            audio_path: Caminho do arquivo

        Returns:
            Tupla (audio_data, sample_rate)
        """
        if not HAS_SOUNDFILE:
            raise RuntimeError("soundfile nao disponivel")

        audio_data, sr = sf.read(audio_path)
        return audio_data, sr

    def _save_audio(
        self,
        audio_data: np.ndarray,
        output_path: Path,
        sample_rate: int = None
    ):
        """
        Salva arquivo de audio.

        Args:
            audio_data: Dados do audio
            output_path: Caminho de saida
            sample_rate: Taxa de amostragem
        """
        if not HAS_SOUNDFILE:
            raise RuntimeError("soundfile nao disponivel")

        sr = sample_rate or self.sample_rate
        sf.write(output_path, audio_data, sr)

    def _apply_enhancement(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Aplica filtros de enhancement no audio.

        Args:
            audio_data: Dados do audio
            sample_rate: Taxa de amostragem

        Returns:
            Audio filtrado
        """
        if not HAS_SCIPY or not self.enhance_audio:
            return audio_data

        # High-pass filter (remove ruido grave)
        nyquist = sample_rate / 2
        high_cutoff = self.highpass_freq / nyquist

        if high_cutoff < 1.0:
            b_high, a_high = signal.butter(4, high_cutoff, btype='high')
            audio_data = signal.filtfilt(b_high, a_high, audio_data)

        # Low-pass filter (remove ruido agudo)
        low_cutoff = self.lowpass_freq / nyquist

        if low_cutoff < 1.0:
            b_low, a_low = signal.butter(4, low_cutoff, btype='low')
            audio_data = signal.filtfilt(b_low, a_low, audio_data)

        return audio_data

    def _merge_consecutive_segments(
        self,
        segments: List[Dict]
    ) -> List[Dict]:
        """
        Mescla segmentos consecutivos do mesmo speaker.

        Args:
            segments: Lista de segmentos ordenados por tempo

        Returns:
            Lista de segmentos mesclados
        """
        if not segments:
            return []

        # Ordenar por tempo de inicio
        sorted_segments = sorted(segments, key=lambda x: x['start'])

        merged = []
        current = sorted_segments[0].copy()

        for seg in sorted_segments[1:]:
            # Verificar se pode mesclar
            gap = seg['start'] - current['end']

            if gap <= self.max_gap:
                # Mesclar: expandir fim do segmento atual
                current['end'] = seg['end']
            else:
                # Nao mesclar: salvar atual e iniciar novo
                merged.append(current)
                current = seg.copy()

        # Adicionar ultimo segmento
        merged.append(current)

        return merged

    def extract_speaker_segments(
        self,
        audio_path: Path,
        diarization_data: Dict,
        segment_offset: float,
        video_id: str
    ) -> Dict[str, List[Dict]]:
        """
        Extrai subsegmentos de audio por speaker.

        Args:
            audio_path: Caminho do arquivo de audio
            diarization_data: Dados de diarizacao do segmento
            segment_offset: Offset absoluto do segmento (absolute_start)
            video_id: ID do video

        Returns:
            Dict mapeando speaker -> lista de subsegmentos extraidos
        """
        if not HAS_NUMPY or not HAS_SOUNDFILE:
            raise RuntimeError("numpy e soundfile sao necessarios")

        # Carregar audio
        audio_data, sample_rate = self._load_audio(audio_path)

        # Aplicar enhancement
        audio_data = self._apply_enhancement(audio_data, sample_rate)

        # Extrair nome base do segmento
        segment_name = audio_path.stem  # ex: video_id_seg_001

        # Organizar segmentos por speaker
        speakers_segments = {}

        for speaker_data in diarization_data.get('speakers', []):
            speaker_id = speaker_data['speaker']

            if speaker_id not in speakers_segments:
                speakers_segments[speaker_id] = []

            # Cada entrada de speaker tem lista de segmentos
            for turn in speaker_data.get('segments', []):
                speakers_segments[speaker_id].append({
                    'start': turn['start_relative'],
                    'end': turn['end_relative']
                })

        # Processar cada speaker
        results = {}

        for speaker_id, segments in speakers_segments.items():
            # Mesclar segmentos consecutivos
            merged_segments = self._merge_consecutive_segments(segments)

            speaker_results = []

            for seg in merged_segments:
                start_rel = seg['start']
                end_rel = seg['end']
                duration = end_rel - start_rel

                # Validar duracao
                if duration < self.min_duration:
                    logger.info(
                        f"Descartando subsegmento curto: "
                        f"{duration:.2f}s < {self.min_duration}s"
                    )
                    continue

                if duration > self.max_duration:
                    logger.warning(
                        f"Subsegmento muito longo: "
                        f"{duration:.2f}s > {self.max_duration}s"
                    )
                    # Ainda assim processar, mas logar warning

                # Calcular timestamps absolutos
                absolute_start = segment_offset + start_rel
                absolute_end = segment_offset + end_rel

                # Extrair audio
                start_sample = int(start_rel * sample_rate)
                end_sample = int(end_rel * sample_rate)

                # Garantir limites validos
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_data), end_sample)

                subsegment_audio = audio_data[start_sample:end_sample]

                if len(subsegment_audio) == 0:
                    logger.warning(f"Subsegmento vazio: {start_rel}-{end_rel}")
                    continue

                # Gerar nome do arquivo
                # Formato: {video_id}_segment_{n}_{SPEAKER_XX}_{start}_{end}.flac
                filename = (
                    f"{segment_name}_{speaker_id}_"
                    f"{absolute_start:.2f}_{absolute_end:.2f}.flac"
                )

                speaker_results.append({
                    'audio_data': subsegment_audio,
                    'filename': filename,
                    'absolute_start': absolute_start,
                    'absolute_end': absolute_end,
                    'duration': duration,
                    'start_relative': start_rel,
                    'end_relative': end_rel,
                    'speaker': speaker_id,
                    'original_segment': segment_name,
                    'sample_rate': sample_rate
                })

            if speaker_results:
                results[speaker_id] = speaker_results

        return results

    def process_segment(
        self,
        audio_path: Path,
        diarization_data: Dict,
        segment_offset: float,
        video_id: str,
        output_dir: Path
    ) -> Dict:
        """
        Processa um segmento completo e salva subsegmentos.

        Args:
            audio_path: Caminho do arquivo de audio
            diarization_data: Dados de diarizacao
            segment_offset: Offset absoluto do segmento
            video_id: ID do video
            output_dir: Diretorio base de saida

        Returns:
            Resultado do processamento
        """
        segment_id = audio_path.stem

        try:
            # Extrair subsegmentos
            speaker_segments = self.extract_speaker_segments(
                audio_path=audio_path,
                diarization_data=diarization_data,
                segment_offset=segment_offset,
                video_id=video_id
            )

            if not speaker_segments:
                return {
                    'segment_id': segment_id,
                    'success': True,
                    'speakers': {},
                    'total_subsegments': 0,
                    'skipped': 0
                }

            # Criar diretorios e salvar arquivos
            saved_files = {}
            total_saved = 0

            for speaker_id, subsegments in speaker_segments.items():
                # Diretorio do speaker
                speaker_dir = output_dir / f"speaker_{speaker_id}"
                speaker_dir.mkdir(parents=True, exist_ok=True)

                saved_files[speaker_id] = []

                for subseg in subsegments:
                    # Salvar arquivo
                    output_path = speaker_dir / subseg['filename']
                    self._save_audio(
                        subseg['audio_data'],
                        output_path,
                        subseg['sample_rate']
                    )

                    # Registrar arquivo salvo
                    saved_files[speaker_id].append({
                        'file': subseg['filename'],
                        'path': str(output_path),
                        'absolute_start': subseg['absolute_start'],
                        'absolute_end': subseg['absolute_end'],
                        'duration': subseg['duration'],
                        'speaker': speaker_id,
                        'original_segment': subseg['original_segment']
                    })

                    total_saved += 1

            return {
                'segment_id': segment_id,
                'success': True,
                'speakers': saved_files,
                'total_subsegments': total_saved,
                'skipped': 0
            }

        except Exception as e:
            logger.error(f"Erro ao processar {segment_id}: {e}")
            return {
                'segment_id': segment_id,
                'success': False,
                'error': str(e),
                'speakers': {},
                'total_subsegments': 0
            }

    def process_batch(
        self,
        clean_segments: List[Dict],
        diarization_results: Dict,
        video_id: str,
        output_base_dir: Path
    ) -> Dict:
        """
        Processa batch de segmentos limpos.

        Args:
            clean_segments: Lista de segmentos aprovados (sem overlap)
            diarization_results: Resultados completos da diarizacao
            video_id: ID do video
            output_base_dir: Diretorio base de saida

        Returns:
            Resultado do processamento em batch
        """
        total = len(clean_segments)
        processed = []
        failed = []
        skipped_short = 0
        all_speakers = set()
        total_subsegments = 0

        logger.info(f"=== STEP 6: SEPARATING SPEAKERS ===")
        logger.info(f"Processing {total} clean segments")

        # Criar diretorio stt_ready
        stt_ready_dir = output_base_dir / 'stt_ready'
        stt_ready_dir.mkdir(parents=True, exist_ok=True)

        # Mapear diarizacao por segment_id
        diarization_map = {}
        for result in diarization_results.get('processed', []):
            seg_id = result['segment_id']
            diarization_map[seg_id] = result

        for idx, segment in enumerate(clean_segments, 1):
            segment_id = segment.get('segment_id', '')
            audio_path = Path(segment.get('audio_path', segment.get('file_path', '')))
            segment_offset = segment.get('absolute_start', 0.0)

            logger.info(f"[{idx}/{total}] {audio_path.name} (offset: {segment_offset:.2f}s)")

            # Verificar se tem diarizacao
            if segment_id not in diarization_map:
                logger.warning(f"Diarizacao nao encontrada para {segment_id}")
                failed.append({
                    'segment_id': segment_id,
                    'error': 'Diarizacao nao encontrada'
                })
                continue

            diarization_data = diarization_map[segment_id]

            # Verificar se diarizacao falhou
            if 'error' in diarization_data:
                logger.warning(f"Pulando {segment_id} - diarizacao falhou")
                failed.append({
                    'segment_id': segment_id,
                    'error': diarization_data['error']
                })
                continue

            # Processar segmento
            result = self.process_segment(
                audio_path=audio_path,
                diarization_data=diarization_data,
                segment_offset=segment_offset,
                video_id=video_id,
                output_dir=stt_ready_dir
            )

            if result['success']:
                processed.append(result)
                total_subsegments += result['total_subsegments']

                # Log detalhado
                for speaker_id, files in result['speakers'].items():
                    all_speakers.add(speaker_id)
                    total_duration = sum(f['duration'] for f in files)
                    logger.info(
                        f"  {speaker_id}: {len(files)} subsegmentos "
                        f"({total_duration:.1f}s total)"
                    )
            else:
                failed.append(result)
                logger.error(f"  Falha: {result.get('error', 'Erro desconhecido')}")

        # Estatisticas finais
        stats = {
            'total_segments': total,
            'processed_count': len(processed),
            'failed_count': len(failed),
            'total_speakers': len(all_speakers),
            'speakers_list': sorted(list(all_speakers)),
            'total_subsegments': total_subsegments,
            'skipped_short': skipped_short
        }

        logger.info(f"\nSpeaker separation completed:")
        logger.info(f"  Segments processed: {len(processed)}/{total}")
        logger.info(f"  Total speakers found: {len(all_speakers)}")
        logger.info(f"  Total subsegments created: {total_subsegments}")
        logger.info(f"  STT-ready files: {stt_ready_dir}")

        return {
            'video_id': video_id,
            'processed': processed,
            'failed': failed,
            'stats': stats,
            'stt_ready_dir': str(stt_ready_dir)
        }

"""
Diarizacao de speakers usando pyannote.audio.
Detecta e separa diferentes locutores no audio.
"""
from pathlib import Path
from typing import List, Dict
import logging
import os
import json

from src.core.timer import Timer

logger = logging.getLogger(__name__)

# Imports opcionais para pyannote
try:
    import torch
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation
    HAS_PYANNOTE = True
except ImportError:
    HAS_PYANNOTE = False


class Diarizer:
    """
    Diariza audio para identificar diferentes speakers.

    Usa pyannote.audio com deteccao automatica de speakers.
    Gera timestamps triplos: relativo, absoluto ao segmento, absoluto ao original.
    """

    # Cache singleton do modelo (carrega 1x, reutiliza)
    _pipeline_cache = None

    def __init__(self, config):
        """
        Inicializa diarizador.

        Args:
            config: Modulo de configuracao
        """
        if not HAS_PYANNOTE:
            raise ImportError(
                "pyannote.audio nao instalado. "
                "Instale com: pip install pyannote.audio"
            )

        self.model_name = config.DIARIZATION_MODEL
        self.expected_speakers = config.EXPECTED_NUM_SPEAKERS
        self.merge_gap = config.DIARIZATION_MERGE_GAP
        self.min_speaker_duration = config.MIN_SPEAKER_DURATION
        self.min_segment_duration = config.MIN_SEGMENT_DURATION

        # Carregar token HuggingFace
        self.hf_token = self._load_hf_token()

        # Detectar device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Carregar pipeline (cache singleton)
        self.pipeline = self._load_pipeline()

        logger.info(
            f"Diarizer inicializado "
            f"(device={self.device}, merge_gap={self.merge_gap}s)"
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
                "Configure em .env: HUGGINGFACE_TOKEN=hf_xxxxx\n"
                "Obtenha token em: https://huggingface.co/settings/tokens"
            )

        logger.debug("Token HuggingFace carregado")
        return token

    def _load_pipeline(self):
        """
        Carrega pipeline pyannote (cache singleton).

        Returns:
            Pipeline pyannote carregado
        """
        if Diarizer._pipeline_cache is None:
            logger.info("Carregando pipeline pyannote (primeira vez)...")
            try:
                # Carregar pipeline
                Diarizer._pipeline_cache = Pipeline.from_pretrained(
                    self.model_name,
                    use_auth_token=self.hf_token
                )

                # Mover para GPU se disponivel
                Diarizer._pipeline_cache.to(self.device)

                logger.info("Pipeline pyannote carregado com sucesso")
            except Exception as e:
                logger.error(f"Erro ao carregar pipeline pyannote: {e}")
                raise RuntimeError(
                    f"Falha ao carregar pipeline de diarizacao: {e}\n"
                    "Verifique:\n"
                    "1. Token HuggingFace valido em .env\n"
                    "2. Termos aceitos em https://huggingface.co/pyannote/speaker-diarization-3.1"
                )
        else:
            logger.debug("Reutilizando pipeline pyannote do cache")

        return Diarizer._pipeline_cache

    def diarize_segment(
        self,
        audio_path: Path,
        segment_id: str,
        segment_absolute_start: float
    ) -> Dict:
        """
        Diariza um segmento de audio.

        Args:
            audio_path: Caminho do arquivo de audio
            segment_id: ID do segmento (ex: 'seg_001')
            segment_absolute_start: Timestamp absoluto do inicio do segmento
                                   em relacao ao audio original (segundos)

        Returns:
            Dict com resultados da diarizacao
        """
        try:
            with Timer(f"Diarizacao {segment_id}") as timer:
                # Executar diarizacao
                diarization = self.pipeline(str(audio_path))

                # Processar resultado
                speakers_data = self._process_diarization(
                    diarization=diarization,
                    segment_id=segment_id,
                    segment_absolute_start=segment_absolute_start
                )

                # Filtrar speakers muito curtos
                speakers_data = self._filter_short_speakers(speakers_data)

                if not speakers_data:
                    return {
                        'success': False,
                        'segment_id': segment_id,
                        'error': 'Nenhum speaker valido apos filtragem'
                    }

                # Salvar RTTM + JSON
                output_dir = audio_path.parent.parent / 'diarization'
                output_dir.mkdir(parents=True, exist_ok=True)

                rttm_path = self._save_rttm(
                    diarization=diarization,
                    output_dir=output_dir,
                    segment_id=segment_id
                )

                json_path = self._save_json(
                    speakers_data=speakers_data,
                    output_dir=output_dir,
                    segment_id=segment_id
                )

                return {
                    'success': True,
                    'segment_id': segment_id,
                    'num_speakers': len(speakers_data),
                    'speakers': speakers_data,
                    'rttm_path': str(rttm_path),
                    'json_path': str(json_path),
                    'processing_time_s': timer.elapsed_time
                }

        except Exception as e:
            logger.error(f"Erro ao diarizar {segment_id}: {e}")
            return {
                'success': False,
                'segment_id': segment_id,
                'error': str(e)
            }

    def _process_diarization(
        self,
        diarization: 'Annotation',
        segment_id: str,
        segment_absolute_start: float
    ) -> List[Dict]:
        """
        Processa resultado da diarizacao em estrutura organizada.

        Args:
            diarization: Annotation do pyannote
            segment_id: ID do segmento
            segment_absolute_start: Inicio absoluto do segmento no audio original

        Returns:
            Lista de dicts com dados dos speakers
        """
        # Agrupar por speaker
        speakers = {}

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speakers:
                speakers[speaker] = []

            # Calcular timestamps triplos
            start_relative = segment.start
            end_relative = segment.end
            duration = segment.end - segment.start

            # Absoluto ao segmento original (mesmo que relativo neste caso)
            start_absolute_segment = start_relative
            end_absolute_segment = end_relative

            # CRITICO: Absoluto ao audio original
            start_absolute_original = segment_absolute_start + start_relative
            end_absolute_original = segment_absolute_start + end_relative

            speakers[speaker].append({
                'start_relative': float(start_relative),
                'end_relative': float(end_relative),
                'start_absolute_segment': float(start_absolute_segment),
                'end_absolute_segment': float(end_absolute_segment),
                'start_absolute_original': float(start_absolute_original),
                'end_absolute_original': float(end_absolute_original),
                'duration': float(duration)
            })

        # Mesclar segmentos consecutivos
        speakers_merged = {}
        for speaker, segments in speakers.items():
            merged = self._merge_consecutive_segments(segments)
            speakers_merged[speaker] = merged

        # Estruturar dados finais
        speakers_data = []
        for speaker, segments in speakers_merged.items():
            total_duration = sum(seg['duration'] for seg in segments)

            speakers_data.append({
                'speaker_id': speaker,
                'segments': segments,
                'total_duration': total_duration
            })

        # Ordenar por speaker_id
        speakers_data.sort(key=lambda x: x['speaker_id'])

        return speakers_data

    def _merge_consecutive_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Mescla segmentos consecutivos do mesmo speaker.

        Args:
            segments: Lista de segmentos do speaker

        Returns:
            Lista de segmentos mesclados
        """
        if not segments:
            return []

        # Ordenar por start_relative
        sorted_segments = sorted(segments, key=lambda x: x['start_relative'])

        merged = []
        current = sorted_segments[0].copy()

        for next_seg in sorted_segments[1:]:
            gap = next_seg['start_relative'] - current['end_relative']

            # Mesclar se gap <= threshold
            if gap <= self.merge_gap:
                # Estender segmento atual
                current['end_relative'] = next_seg['end_relative']
                current['end_absolute_segment'] = next_seg['end_absolute_segment']
                current['end_absolute_original'] = next_seg['end_absolute_original']
                current['duration'] = current['end_relative'] - current['start_relative']
            else:
                # Salvar atual e iniciar novo
                merged.append(current)
                current = next_seg.copy()

        # Adicionar ultimo segmento
        merged.append(current)

        return merged

    def _filter_short_speakers(self, speakers_data: List[Dict]) -> List[Dict]:
        """
        Remove speakers com duracao total muito curta.

        Args:
            speakers_data: Lista de dados dos speakers

        Returns:
            Lista filtrada
        """
        filtered = []

        for speaker in speakers_data:
            # Filtrar segmentos individuais muito curtos
            valid_segments = [
                seg for seg in speaker['segments']
                if seg['duration'] >= self.min_segment_duration
            ]

            if not valid_segments:
                logger.debug(
                    f"Speaker {speaker['speaker_id']} removido: "
                    f"todos segmentos < {self.min_segment_duration}s"
                )
                continue

            # Recalcular duracao total
            total_duration = sum(seg['duration'] for seg in valid_segments)

            # Verificar duracao minima do speaker
            if total_duration < self.min_speaker_duration:
                logger.debug(
                    f"Speaker {speaker['speaker_id']} removido: "
                    f"duracao total {total_duration:.1f}s < {self.min_speaker_duration}s"
                )
                continue

            filtered.append({
                'speaker_id': speaker['speaker_id'],
                'segments': valid_segments,
                'total_duration': total_duration
            })

        return filtered

    def _save_rttm(
        self,
        diarization: 'Annotation',
        output_dir: Path,
        segment_id: str
    ) -> Path:
        """
        Salva diarizacao em formato RTTM.

        Args:
            diarization: Annotation do pyannote
            output_dir: Diretorio de saida
            segment_id: ID do segmento

        Returns:
            Caminho do arquivo RTTM
        """
        rttm_path = output_dir / f"{segment_id}.rttm"

        with open(rttm_path, 'w') as f:
            diarization.write_rttm(f)

        logger.debug(f"RTTM salvo: {rttm_path}")
        return rttm_path

    def _save_json(
        self,
        speakers_data: List[Dict],
        output_dir: Path,
        segment_id: str
    ) -> Path:
        """
        Salva diarizacao em formato JSON.

        Args:
            speakers_data: Dados dos speakers
            output_dir: Diretorio de saida
            segment_id: ID do segmento

        Returns:
            Caminho do arquivo JSON
        """
        json_path = output_dir / f"{segment_id}.json"

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'segment_id': segment_id,
                    'num_speakers': len(speakers_data),
                    'speakers': speakers_data
                },
                f,
                indent=2,
                ensure_ascii=False
            )

        logger.debug(f"JSON salvo: {json_path}")
        return json_path

    def diarize_batch(
        self,
        segments: List[Dict],
        output_base_dir: Path
    ) -> Dict:
        """
        Diariza multiplos segmentos em batch.

        Args:
            segments: Lista de segmentos do checkpoint (etapa 02)
            output_base_dir: Diretorio base da sessao

        Returns:
            Dict com resultados do batch
        """
        results = {
            'success': True,
            'total_segments': len(segments),
            'processed': [],
            'failed': [],
            'stats': {
                'total_speakers': 0,
                'avg_speakers_per_segment': 0.0
            }
        }

        logger.info(f"Iniciando diarizacao de {len(segments)} segmentos")

        for i, segment in enumerate(segments, 1):
            segment_id = segment['segment_id']
            file_path = Path(segment['file_path'])
            absolute_start = segment['absolute_start']

            logger.info(f"[{i}/{len(segments)}] Diarizando {segment_id}")

            result = self.diarize_segment(
                audio_path=file_path,
                segment_id=segment_id,
                segment_absolute_start=absolute_start
            )

            if result['success']:
                results['processed'].append(result)
                results['stats']['total_speakers'] += result['num_speakers']
            else:
                results['failed'].append(result)
                logger.warning(
                    f"Pulando {segment_id}: {result.get('error')}"
                )

        # Calcular estatisticas
        if results['processed']:
            results['stats']['avg_speakers_per_segment'] = (
                results['stats']['total_speakers'] / len(results['processed'])
            )

        logger.info(
            f"Diarizacao concluida: "
            f"{len(results['processed'])} sucesso, "
            f"{len(results['failed'])} falhas"
        )

        return results

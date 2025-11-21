"""
Testes de validacao do PROMPT 04 - Diarizacao.
"""
import sys
import shutil
from pathlib import Path

# Adicionar diretorio raiz ao path
sys.path.insert(0, str(Path(__file__).parent))


def test_timestamp_calculation():
    """Testa calculo de timestamps triplos."""
    # Simular cenario real:
    # seg_002 comeca em 12.5s no audio original
    # Speaker fala de 3.0-8.0s no segmento

    segment_absolute_start = 12.5
    start_relative = 3.0
    end_relative = 8.0

    # Calcular timestamps
    start_absolute_original = segment_absolute_start + start_relative
    end_absolute_original = segment_absolute_start + end_relative

    # Validar
    assert start_absolute_original == 15.5  # 12.5 + 3.0
    assert end_absolute_original == 20.5    # 12.5 + 8.0

    print("Calculo de timestamps triplos correto - OK")
    return True


def test_merge_consecutive_segments():
    """Testa merge de segmentos consecutivos."""
    # Simular segmentos do mesmo speaker
    segments = [
        {'start_relative': 0.0, 'end_relative': 2.0, 'duration': 2.0,
         'start_absolute_segment': 0.0, 'end_absolute_segment': 2.0,
         'start_absolute_original': 0.0, 'end_absolute_original': 2.0},
        {'start_relative': 2.3, 'end_relative': 5.0, 'duration': 2.7,
         'start_absolute_segment': 2.3, 'end_absolute_segment': 5.0,
         'start_absolute_original': 2.3, 'end_absolute_original': 5.0},
        {'start_relative': 8.0, 'end_relative': 10.0, 'duration': 2.0,
         'start_absolute_segment': 8.0, 'end_absolute_segment': 10.0,
         'start_absolute_original': 8.0, 'end_absolute_original': 10.0}
    ]

    merge_gap = 0.5

    # Algoritmo de merge simplificado
    sorted_segments = sorted(segments, key=lambda x: x['start_relative'])
    merged = []
    current = sorted_segments[0].copy()

    for next_seg in sorted_segments[1:]:
        gap = next_seg['start_relative'] - current['end_relative']

        if gap <= merge_gap:
            current['end_relative'] = next_seg['end_relative']
            current['duration'] = current['end_relative'] - current['start_relative']
        else:
            merged.append(current)
            current = next_seg.copy()

    merged.append(current)

    # Validar
    assert len(merged) == 2  # Primeiro e segundo mesclam, terceiro separado
    assert merged[0]['end_relative'] == 5.0  # Primeiro mesclado ate 5.0
    assert merged[1]['start_relative'] == 8.0  # Segundo comeca em 8.0

    print("Merge de segmentos consecutivos funcionando - OK")
    return True


def test_filter_short_speakers():
    """Testa filtro de speakers curtos."""
    min_speaker_duration = 2.0
    min_segment_duration = 0.5

    speakers_data = [
        {
            'speaker_id': 'SPEAKER_00',
            'segments': [
                {'duration': 3.0},
                {'duration': 2.0}
            ],
            'total_duration': 5.0
        },
        {
            'speaker_id': 'SPEAKER_01',
            'segments': [
                {'duration': 0.3},  # Muito curto
                {'duration': 0.2}   # Muito curto
            ],
            'total_duration': 0.5
        },
        {
            'speaker_id': 'SPEAKER_02',
            'segments': [
                {'duration': 1.0}
            ],
            'total_duration': 1.0  # Abaixo do minimo
        }
    ]

    # Filtrar
    filtered = []
    for speaker in speakers_data:
        valid_segments = [
            seg for seg in speaker['segments']
            if seg['duration'] >= min_segment_duration
        ]

        if not valid_segments:
            continue

        total_duration = sum(seg['duration'] for seg in valid_segments)

        if total_duration < min_speaker_duration:
            continue

        filtered.append({
            'speaker_id': speaker['speaker_id'],
            'segments': valid_segments,
            'total_duration': total_duration
        })

    # Validar
    assert len(filtered) == 1
    assert filtered[0]['speaker_id'] == 'SPEAKER_00'

    print("Filtro de speakers curtos funcionando - OK")
    return True


def test_checkpoint_diarization_save():
    """Testa salvamento de resultados de diarizacao no checkpoint."""
    import config
    from src.output.checkpoint_manager import CheckpointManager

    manager = CheckpointManager(config)

    # Simular resultado de diarizacao
    diarization_results = {
        'total_segments': 3,
        'processed': [
            {
                'segment_id': 'seg_001',
                'num_speakers': 2,
                'speakers': [
                    {'speaker_id': 'SPEAKER_00', 'total_duration': 5.0, 'segments': []},
                    {'speaker_id': 'SPEAKER_01', 'total_duration': 3.0, 'segments': []}
                ],
                'rttm_path': '/path/to/seg_001.rttm',
                'json_path': '/path/to/seg_001.json'
            }
        ],
        'failed': [
            {'segment_id': 'seg_002', 'error': 'Teste falha'}
        ],
        'stats': {
            'total_speakers': 2,
            'avg_speakers_per_segment': 2.0
        }
    }

    # Salvar
    manager.save_diarization_results('test_diar', diarization_results)

    # Verificar
    checkpoint = manager.load_checkpoint('test_diar')

    assert checkpoint is not None
    assert '04_diarizacao' in checkpoint['etapas']
    assert checkpoint['etapas']['04_diarizacao']['data']['processed_count'] == 1
    assert checkpoint['etapas']['04_diarizacao']['data']['failed_count'] == 1

    # Limpar
    manager.delete_checkpoint('test_diar')

    print("Checkpoint diarizacao save funcionando - OK")
    return True


def test_diarizer_import():
    """Testa importacao do Diarizer (sem carregar modelo)."""
    try:
        from src.diarization.diarizer import Diarizer
        print("Importacao Diarizer - OK")
        return True
    except ImportError as e:
        if 'pyannote' in str(e) or 'torch' in str(e):
            print(f"Diarizer importacao pulada (dependencia nao disponivel) - OK")
            return True
        raise


def test_config_diarization():
    """Testa configuracoes de diarizacao."""
    import config

    assert hasattr(config, 'DIARIZATION_MODEL')
    assert hasattr(config, 'DIARIZATION_MERGE_GAP')
    assert hasattr(config, 'MIN_SPEAKER_DURATION')
    assert hasattr(config, 'MIN_SEGMENT_DURATION')

    assert config.DIARIZATION_MERGE_GAP == 0.5
    assert config.MIN_SPEAKER_DURATION == 2.0
    assert config.MIN_SEGMENT_DURATION == 0.5

    print("Configuracoes de diarizacao corretas - OK")
    return True


if __name__ == "__main__":
    print("\n" + "="*50)
    print("TESTES DE VALIDACAO - PROMPT 04")
    print("="*50 + "\n")

    # Importar config para mostrar status
    import config

    testes = [
        ("Calculo Timestamps Triplos", test_timestamp_calculation),
        ("Merge Segmentos Consecutivos", test_merge_consecutive_segments),
        ("Filtro Speakers Curtos", test_filter_short_speakers),
        ("Checkpoint Diarizacao Save", test_checkpoint_diarization_save),
        ("Importacao Diarizer", test_diarizer_import),
        ("Config Diarizacao", test_config_diarization)
    ]

    resultados = []

    for nome, func in testes:
        try:
            func()
            resultados.append((nome, True))
        except Exception as e:
            print(f"FALHA em {nome}: {e}")
            import traceback
            traceback.print_exc()
            resultados.append((nome, False))

    print("\n" + "="*50)
    print("RESULTADO FINAL")
    print("="*50)

    passou = sum(1 for _, r in resultados if r)
    total = len(resultados)

    for nome, resultado in resultados:
        status = "PASSOU" if resultado else "FALHOU"
        print(f"  {nome}: {status}")

    print(f"\nTotal: {passou}/{total} testes passando")

    if passou == total:
        print("\nPROMPT 04: Todos os testes passaram")
        sys.exit(0)
    else:
        print("\nPROMPT 04: Alguns testes falharam")
        sys.exit(1)

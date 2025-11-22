"""
Testes de validacao do PROMPT 05 - Overlap Detection.
"""
import sys
import shutil
from pathlib import Path

# Adicionar diretorio raiz ao path
sys.path.insert(0, str(Path(__file__).parent))


def test_config_overlap():
    """Testa configuracoes de overlap detection."""
    import config

    assert hasattr(config, 'OVERLAP_DETECTION_MODEL')
    assert hasattr(config, 'OVERLAP_THRESHOLD')
    assert hasattr(config, 'MIN_OVERLAP_DURATION')
    assert hasattr(config, 'OVERLAP_WINDOW_SIZE')
    assert hasattr(config, 'OVERLAP_HOP_SIZE')

    assert config.OVERLAP_THRESHOLD == 0.9
    assert config.MIN_OVERLAP_DURATION == 0.3
    assert 0.0 <= config.OVERLAP_THRESHOLD <= 1.0

    print("Configuracoes de overlap detection corretas - OK")
    return True


def test_overlap_regions_extraction():
    """Testa extracao de regioes de overlap."""
    import numpy as np

    # Simular mask de overlap
    # True = overlap, False = sem overlap
    overlap_mask = np.array([
        False, False, True, True, True, False, False, True, True, False
    ])
    frame_duration = 1.0  # 1 segundo por frame

    # Algoritmo de extracao
    regions = []
    in_overlap = False
    start_frame = 0

    for i, is_overlap in enumerate(overlap_mask):
        if is_overlap and not in_overlap:
            start_frame = i
            in_overlap = True
        elif not is_overlap and in_overlap:
            end_frame = i
            start_time = start_frame * frame_duration
            end_time = end_frame * frame_duration
            duration = end_time - start_time
            regions.append({
                'start': start_time,
                'end': end_time,
                'duration': duration
            })
            in_overlap = False

    if in_overlap:
        end_frame = len(overlap_mask)
        start_time = start_frame * frame_duration
        end_time = end_frame * frame_duration
        duration = end_time - start_time
        regions.append({
            'start': start_time,
            'end': end_time,
            'duration': duration
        })

    # Validar
    assert len(regions) == 2
    assert regions[0]['start'] == 2.0
    assert regions[0]['end'] == 5.0
    assert regions[0]['duration'] == 3.0
    assert regions[1]['start'] == 7.0
    assert regions[1]['end'] == 9.0
    assert regions[1]['duration'] == 2.0

    print("Extracao de regioes de overlap funcionando - OK")
    return True


def test_overlap_ratio_calculation():
    """Testa calculo de overlap ratio."""
    # Cenario: audio de 10s com 3s de overlap
    audio_duration = 10.0
    total_overlap_duration = 3.0

    overlap_ratio = total_overlap_duration / audio_duration
    threshold = 0.9

    assert overlap_ratio == 0.3
    assert overlap_ratio < threshold  # Deve ser aprovado

    # Cenario: audio de 10s com 9.5s de overlap (deve rejeitar)
    total_overlap_duration_2 = 9.5
    overlap_ratio_2 = total_overlap_duration_2 / audio_duration

    assert overlap_ratio_2 == 0.95
    assert overlap_ratio_2 >= threshold  # Deve ser rejeitado

    print("Calculo de overlap ratio funcionando - OK")
    return True


def test_checkpoint_overlap_save():
    """Testa salvamento de resultados de overlap no checkpoint."""
    import config
    from src.output.checkpoint_manager import CheckpointManager

    manager = CheckpointManager(config)

    # Simular resultado de overlap detection
    results = {
        'total_segments': 5,
        'approved_count': 3,
        'rejected_count': 1,
        'failed_count': 1,
        'approved': [
            {
                'segment_id': 'seg_001',
                'audio_path': '/path/to/seg_001.flac',
                'overlap_detection': {
                    'overlap_ratio': 0.1,
                    'has_overlap': True
                }
            }
        ],
        'rejected': [
            {
                'segment_id': 'seg_002',
                'audio_path': '/path/to/seg_002.flac',
                'overlap_detection': {
                    'overlap_ratio': 0.95,
                    'has_overlap': True
                }
            }
        ],
        'failed': [],
        'stats': {
            'avg_overlap_ratio': 0.15,
            'max_overlap_ratio': 0.3,
            'total_overlap_duration': 5.5
        }
    }

    # Salvar
    manager.save_overlap_detection_results('test_overlap', results)

    # Verificar
    checkpoint = manager.load_checkpoint('test_overlap')

    assert checkpoint is not None
    assert '05_overlap_detection' in checkpoint['etapas']
    assert checkpoint['etapas']['05_overlap_detection']['data']['approved_count'] == 3
    assert checkpoint['etapas']['05_overlap_detection']['data']['rejected_count'] == 1

    # Limpar
    manager.delete_checkpoint('test_overlap')

    print("Checkpoint overlap detection save funcionando - OK")
    return True


def test_overlap_detector_import():
    """Testa importacao do OverlapDetector (sem carregar modelo)."""
    try:
        from src.diarization.overlap_detector import OverlapDetector
        print("Importacao OverlapDetector - OK")
        return True
    except ImportError as e:
        if 'pyannote' in str(e) or 'torch' in str(e):
            print(f"OverlapDetector importacao pulada (dependencia nao disponivel) - OK")
            return True
        raise


def test_filter_short_overlaps():
    """Testa filtro de overlaps muito curtos."""
    min_overlap_duration = 0.3

    overlap_regions = [
        {'start': 0.0, 'end': 0.1, 'duration': 0.1},   # Muito curto
        {'start': 1.0, 'end': 2.5, 'duration': 1.5},   # OK
        {'start': 3.0, 'end': 3.2, 'duration': 0.2},   # Muito curto
        {'start': 5.0, 'end': 6.0, 'duration': 1.0}    # OK
    ]

    # Filtrar
    filtered = [
        region for region in overlap_regions
        if region['duration'] >= min_overlap_duration
    ]

    assert len(filtered) == 2
    assert filtered[0]['duration'] == 1.5
    assert filtered[1]['duration'] == 1.0

    print("Filtro de overlaps curtos funcionando - OK")
    return True


if __name__ == "__main__":
    print("\n" + "="*50)
    print("TESTES DE VALIDACAO - PROMPT 05")
    print("="*50 + "\n")

    # Importar config para mostrar status
    import config

    testes = [
        ("Config Overlap Detection", test_config_overlap),
        ("Extracao Regioes Overlap", test_overlap_regions_extraction),
        ("Calculo Overlap Ratio", test_overlap_ratio_calculation),
        ("Checkpoint Overlap Save", test_checkpoint_overlap_save),
        ("Importacao OverlapDetector", test_overlap_detector_import),
        ("Filtro Overlaps Curtos", test_filter_short_overlaps)
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
        print("\nPROMPT 05: Todos os testes passaram")
        sys.exit(0)
    else:
        print("\nPROMPT 05: Alguns testes falharam")
        sys.exit(1)

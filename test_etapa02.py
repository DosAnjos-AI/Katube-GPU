"""
Testes de validacao do PROMPT 02 - Segmentacao Inteligente.
"""
import sys
import shutil
from pathlib import Path

# Adicionar diretorio raiz ao path
sys.path.insert(0, str(Path(__file__).parent))


def test_audio_validation():
    """Testa validacao de duracao do audio original."""
    import numpy as np
    import soundfile as sf
    import config
    from src.preprocessing.audio_segmenter import AudioSegmenter

    segmenter = AudioSegmenter(config)
    test_dir = Path("temp/test_validation")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Teste 1: Audio muito curto (<10s)
    short_audio = np.zeros(5 * 24000)  # 5s
    short_path = test_dir / "short.flac"
    sf.write(short_path, short_audio, 24000)

    result = segmenter.validate_audio_duration(short_path)
    assert not result['valid'], "Audio curto deveria ser rejeitado"
    assert "muito curto" in result['error'].lower()

    # Teste 2: Audio OK (60s)
    ok_audio = np.zeros(60 * 24000)
    ok_path = test_dir / "ok.flac"
    sf.write(ok_path, ok_audio, 24000)

    result = segmenter.validate_audio_duration(ok_path)
    assert result['valid'], f"Audio OK rejeitado: {result.get('error')}"
    assert result['duration'] == 60.0

    # Limpar
    shutil.rmtree(test_dir)

    print("Validacao de audio funcionando - OK")
    return True


def test_segmentation():
    """Testa segmentacao de audio."""
    import numpy as np
    import soundfile as sf
    import config
    from src.preprocessing.audio_segmenter import AudioSegmenter

    segmenter = AudioSegmenter(config)
    test_dir = Path("temp/test_segmentation")
    output_dir = test_dir / "segments"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Criar audio teste: 60s de "fala" simulada
    # (tons alternados para simular fala/silencio)
    sr = 24000
    duration = 60  # segundos

    # Simular fala com pausas
    audio = []
    for i in range(0, duration, 5):  # Blocos de 5s
        # 4s de "fala" (tom)
        t = np.linspace(0, 4, 4 * sr)
        speech = 0.5 * np.sin(2 * np.pi * 440 * t)
        # 1s de silencio
        silence = np.zeros(sr)
        audio.extend(speech)
        audio.extend(silence)

    audio = np.array(audio[:duration * sr])  # Truncar para 60s exatos

    input_path = test_dir / "test_audio.flac"
    sf.write(input_path, audio, sr)

    # Segmentar
    result = segmenter.segment_audio(
        audio_path=input_path,
        output_dir=output_dir,
        video_id="test123"
    )

    assert result['success'], f"Segmentacao falhou: {result.get('error')}"
    assert result['total_segments'] > 0, "Nenhum segmento criado"

    # Validar nomenclatura
    first_segment = result['segments'][0]
    assert first_segment['segment_id'] == 'seg_001'
    assert 'test123_seg_001.flac' in str(first_segment['file_path'])

    # Validar timestamps
    assert first_segment['absolute_start'] >= 0.0
    assert first_segment['absolute_end'] > first_segment['absolute_start']
    assert first_segment['duration'] > 0.0

    # Limpar
    shutil.rmtree(test_dir)

    print(f"Segmentacao funcionando ({result['total_segments']} segmentos) - OK")
    return True


def test_checkpoint_manager():
    """Testa salvamento/carregamento de checkpoints."""
    from src.output.checkpoint_manager import CheckpointManager
    import config

    manager = CheckpointManager(config)

    # Salvar checkpoint
    test_data = {
        'total_segments': 42,
        'processing_time_s': 123.45
    }

    manager.save_checkpoint(
        video_id='test123',
        etapa='02_segmentacao',
        data=test_data
    )

    # Carregar checkpoint
    checkpoint = manager.load_checkpoint('test123')

    assert checkpoint is not None
    assert '02_segmentacao' in checkpoint['etapas']
    assert checkpoint['etapas']['02_segmentacao']['data']['total_segments'] == 42

    # Limpar
    checkpoint_path = Path("dataset/checkpoints/test123.json")
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("CheckpointManager funcionando - OK")
    return True


if __name__ == "__main__":
    print("\n" + "="*50)
    print("TESTES DE VALIDACAO - PROMPT 02")
    print("="*50 + "\n")

    testes = [
        ("Validacao de Audio", test_audio_validation),
        ("Segmentacao de Audio", test_segmentation),
        ("Checkpoint Manager", test_checkpoint_manager)
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
        print("\nPROMPT 02: Todos os testes passaram")
        sys.exit(0)
    else:
        print("\nPROMPT 02: Alguns testes falharam")
        sys.exit(1)

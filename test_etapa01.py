"""
Testes de validacao do PROMPT 01 - Normalizacao Inicial.
"""
import sys
import shutil
from pathlib import Path

# Adicionar diretorio raiz ao path
sys.path.insert(0, str(Path(__file__).parent))


def test_extract_video_id():
    """Testa extracao de video_id."""
    from src.core.input_detector import InputDetector

    casos = [
        ("abc123.flac", "abc123"),
        ("abc123_titulo.flac", "abc123"),
        ("xyz789_nome_longo_aqui.flac", "xyz789"),
        ("def456.mp3", "def456")
    ]

    for filename, expected_id in casos:
        video_id = InputDetector.extract_video_id(filename)
        assert video_id == expected_id, \
            f"Esperado {expected_id}, obtido {video_id}"

    print("Extracao de video_id funcionando - OK")
    return True


def test_audio_normalizer():
    """Testa normalizacao de audio."""
    import numpy as np
    import soundfile as sf
    import config
    from src.preprocessing.audio_normalizer import AudioNormalizer

    # Criar audio teste (silencio de 1s)
    test_dir = Path("temp/test_normalization")
    test_dir.mkdir(parents=True, exist_ok=True)

    input_path = test_dir / "test_input.wav"
    output_path = test_dir / "test_output.flac"

    # Gerar silencio 1s, 44100Hz, stereo
    audio = np.zeros((44100, 2))
    sf.write(input_path, audio, 44100)

    # Normalizar
    normalizer = AudioNormalizer(config)
    result = normalizer.normalize(input_path, output_path)

    assert result['success'], f"Normalizacao falhou: {result.get('error')}"
    assert output_path.exists(), "Arquivo de saida nao foi criado"
    assert result['method'] in ['ffmpeg', 'librosa']

    # Validar propriedades do audio
    info = sf.info(output_path)
    assert info.samplerate == 24000, f"Sample rate incorreto: {info.samplerate}"
    assert info.channels == 1, f"Canais incorretos: {info.channels}"

    # Limpar
    shutil.rmtree(test_dir)

    print(f"Normalizacao funcionando (metodo: {result['method']}) - OK")
    return True


def test_input_detector_subfolders():
    """Testa deteccao de cenario com subpastas."""
    import numpy as np
    import soundfile as sf
    from src.core.input_detector import InputDetector

    # Criar estrutura de teste
    test_dir = Path("temp/test_input_subfolders")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Criar subpasta com audio
    video_dir = test_dir / "abc123_titulo"
    video_dir.mkdir(exist_ok=True)

    audio_path = video_dir / "abc123_titulo.flac"
    audio = np.zeros(24000)  # 1s de silencio
    sf.write(audio_path, audio, 24000)

    # Detectar
    scenario, files = InputDetector.detect_scenario(test_dir, "flac")

    assert scenario == "subfolders"
    assert len(files) == 1
    assert files[0][0] == "abc123"
    assert files[0][1] == audio_path

    # Limpar
    shutil.rmtree(test_dir)

    print("Deteccao de subpastas funcionando - OK")
    return True


def test_input_detector_flat():
    """Testa deteccao de cenario flat."""
    import numpy as np
    import soundfile as sf
    from src.core.input_detector import InputDetector

    # Criar estrutura de teste
    test_dir = Path("temp/test_input_flat")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Criar arquivos na raiz
    audio_path = test_dir / "xyz789_video.flac"
    audio = np.zeros(24000)  # 1s de silencio
    sf.write(audio_path, audio, 24000)

    # Detectar
    scenario, files = InputDetector.detect_scenario(test_dir, "flac")

    assert scenario == "flat"
    assert len(files) == 1
    assert files[0][0] == "xyz789"

    # Limpar
    shutil.rmtree(test_dir)

    print("Deteccao flat funcionando - OK")
    return True


if __name__ == "__main__":
    print("\n" + "="*50)
    print("TESTES DE VALIDACAO - PROMPT 01")
    print("="*50 + "\n")

    testes = [
        ("Extracao de Video ID", test_extract_video_id),
        ("Normalizacao de Audio", test_audio_normalizer),
        ("Deteccao Subpastas", test_input_detector_subfolders),
        ("Deteccao Flat", test_input_detector_flat)
    ]

    resultados = []

    for nome, func in testes:
        try:
            func()
            resultados.append((nome, True))
        except Exception as e:
            print(f"FALHA em {nome}: {e}")
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
        print("\nPROMPT 01: Todos os testes passaram")
        sys.exit(0)
    else:
        print("\nPROMPT 01: Alguns testes falharam")
        sys.exit(1)

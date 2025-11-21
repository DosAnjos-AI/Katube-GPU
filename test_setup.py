"""
Testes de validacao do PROMPT 00 - Setup Base.
"""
import sys
from pathlib import Path

def test_estrutura_diretorios():
    """Valida que todos os diretorios foram criados."""
    dirs_esperados = [
        "audios",
        "dataset/audio_dataset",
        "dataset/historico_dataset",
        "dataset/checkpoints",
        "dataset/logs",
        "temp",
        "src/core",
        "src/preprocessing",
        "src/quality",
        "src/diarization",
        "src/transcription",
        "src/postprocessing",
        "src/output",
        "src/pipeline"
    ]

    for dir_path in dirs_esperados:
        full_path = Path(dir_path)
        assert full_path.exists(), f"Diretorio ausente: {dir_path}"

    print("Todos os diretorios criados - OK")
    return True

def test_config_importavel():
    """Valida que config.py e importavel."""
    try:
        import config
        assert hasattr(config, 'AUDIO_INPUT_DIR'), "AUDIO_INPUT_DIR ausente"
        assert hasattr(config, 'USE_GPU'), "USE_GPU ausente"
        assert hasattr(config, 'PREPROCESSING_SAMPLE_RATE'), "PREPROCESSING_SAMPLE_RATE ausente"
        assert hasattr(config, 'HUGGINGFACE_TOKEN'), "HUGGINGFACE_TOKEN ausente"
        assert hasattr(config, 'WHISPER_MODEL'), "WHISPER_MODEL ausente"
        assert hasattr(config, 'WAV2VEC2_MODEL'), "WAV2VEC2_MODEL ausente"
        print("config.py importavel e valido - OK")
        return True
    except Exception as e:
        raise AssertionError(f"Erro ao importar config: {e}")

def test_src_importavel():
    """Valida que src e importavel."""
    try:
        import src
        assert hasattr(src, '__version__'), "__version__ ausente"
        print(f"src importavel (versao {src.__version__}) - OK")
        return True
    except Exception as e:
        raise AssertionError(f"Erro ao importar src: {e}")

def test_arquivos_base():
    """Valida que arquivos base existem."""
    arquivos = [
        "config.py",
        "run.py",
        "README.md",
        "requirements.txt",
        ".gitignore"
    ]

    for arquivo in arquivos:
        full_path = Path(arquivo)
        assert full_path.exists(), f"Arquivo ausente: {arquivo}"

    print("Todos os arquivos base existem - OK")
    return True

if __name__ == "__main__":
    print("\n" + "="*50)
    print("TESTES DE VALIDACAO - PROMPT 00")
    print("="*50 + "\n")

    testes = [
        ("Estrutura de Diretorios", test_estrutura_diretorios),
        ("Config Importavel", test_config_importavel),
        ("Src Importavel", test_src_importavel),
        ("Arquivos Base", test_arquivos_base)
    ]

    resultados = []

    for nome, func in testes:
        try:
            func()
            resultados.append((nome, True))
        except AssertionError as e:
            print(f"FALHA: {e}")
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
        print("\nPROMPT 00: Todos os testes passaram")
        sys.exit(0)
    else:
        print("\nPROMPT 00: Alguns testes falharam")
        sys.exit(1)

"""
Timer para medir tempo de processamento de cada etapa.
"""
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Timer:
    """
    Context manager para medir tempo de processamento.

    Uso:
        with Timer("Etapa 03") as timer:
            # processar...
        print(timer.elapsed_time)
    """

    def __init__(self, name: str):
        """
        Inicializa timer.

        Args:
            name: Nome da etapa/operacao
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None

    def __enter__(self):
        """Inicia timer."""
        self.start_time = time.time()
        logger.info(f"[{self.name}] Iniciando...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finaliza timer."""
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

        if exc_type is None:
            logger.info(
                f"[{self.name}] Concluido em {self.elapsed_time:.2f}s"
            )
        else:
            logger.error(
                f"[{self.name}] Falhou apos {self.elapsed_time:.2f}s"
            )

        return False  # Nao suprimir excecoes

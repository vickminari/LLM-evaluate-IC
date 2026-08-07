"""
Pacote de chunkerização sintética para a IC.
Oferece chunkers especializados para documentos normativos (por artigo) e não-normativos (por subseção temática).
"""

from .normative_chunker import chunk_normative_document
from .non_normative_chunker import chunk_non_normative_document

__all__ = ["chunk_normative_document", "chunk_non_normative_document"]

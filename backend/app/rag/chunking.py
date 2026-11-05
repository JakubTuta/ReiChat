import typing

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


class TextChunk(typing.NamedTuple):
    content: str
    char_start: int


def chunk_text(text: str) -> list[TextChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )

    return [
        TextChunk(content=doc.page_content, char_start=doc.metadata["start_index"])
        for doc in splitter.create_documents([text])
        if doc.page_content.strip()
    ]

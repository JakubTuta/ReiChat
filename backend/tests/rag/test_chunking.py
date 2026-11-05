from app.rag.chunking import chunk_text


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("") == []


def test_chunk_text_whitespace_only_returns_no_chunks():
    assert chunk_text("   \n\n   ") == []


def test_chunk_text_short_text_is_a_single_chunk():
    chunks = chunk_text("Just a short sentence.")

    assert len(chunks) == 1
    assert chunks[0].content == "Just a short sentence."
    assert chunks[0].char_start == 0


def test_chunk_text_splits_long_text_and_char_start_matches_source():
    text = "word " * 500  # well past CHUNK_SIZE

    chunks = chunk_text(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_start + len(chunk.content)] == chunk.content

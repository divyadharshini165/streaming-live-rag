"""Simulates live speech from a transcript: fixed-size word chunks with arrival timestamps.

A chunk's timestamp is the moment its last word has been spoken (words_per_second).
The utterance end is signalled after a short endpointing silence.
"""


def chunk_transcript(text: str, words_per_chunk: int = 4, words_per_second: float = 2.8,
                     endpoint_silence_s: float = 0.3):
    words = text.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        part = words[i:i + words_per_chunk]
        chunks.append((round((i + len(part)) / words_per_second, 2), " ".join(part)))
    end_t = round((chunks[-1][0] if chunks else 0.0) + endpoint_silence_s, 2)
    return chunks, end_t

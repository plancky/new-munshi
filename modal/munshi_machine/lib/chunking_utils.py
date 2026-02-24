import re
from typing import List, Tuple, Optional


class RecursiveCharacterSplitter:
    """
    Splits text recursively by trying to split on a list of separators in order.
    
    This strategy preserves semantic structure (paragraphs, sentences) better than
    naive fixed-size chunking. It attempts to keep related text together and
    only splits at smaller granularities when necessary to fit within the chunk size.
    
    The hierarchy of splitting is typically:
    1. Paragraphs (\n\n)
    2. Newlines (\n)
    3. Sentences (. ! ?)
    4. Words (space)
    5. Characters (empty string) - last resort
    """
    
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
    ):
        """
        Initialize the splitter.
        
        Args:
            chunk_size: Maximum size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
            separators: List of separators to try splitting on, in priority order.
                       Defaults to ["\n\n", "\n", " ", ""]
            keep_separator: Whether to keep the separator in the chunk (e.g. keep the period at end of sentence)
            is_separator_regex: Whether the separators are regex patterns
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._keep_separator = keep_separator
        self._is_separator_regex = is_separator_regex
        self._separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        final_chunks = []
        
        # Get appropriate separator to use
        separator = self._separators[-1]
        new_separators = []
        
        for i, _s in enumerate(self._separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = self._separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        
        # Perform the split
        splits = self._split_on_separator(text, _separator)

        # Now go through splits, merge small ones, and recurse on large ones
        _good_splits = []
        
        for s in splits:
            if len(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                # If existing accumulation, flush it first
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                
                # Recurse on this large chunk if we have more separators
                if not new_separators:
                    # No more separators, forced to take it as is (or hard split if we implemented that)
                    final_chunks.append(s)
                else:
                    other_splitter = RecursiveCharacterSplitter(
                        chunk_size=self._chunk_size,
                        chunk_overlap=self._chunk_overlap,
                        separators=new_separators,
                        keep_separator=self._keep_separator,
                        is_separator_regex=self._is_separator_regex,
                    )
                    final_chunks.extend(other_splitter.split_text(s))
                    
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, separator)
            final_chunks.extend(merged_text)
            
        return final_chunks

    def _split_on_separator(self, text: str, separator_regex: str) -> List[str]:
        """Split text based on separator, handling keep_separator logic."""
        if separator_regex == "":
            return list(text)
            
        if self._keep_separator:
            # Split and keep separator attached to the preceding segment
            # The regex pattern (?<=...) is a lookbehind assertion
            # But simpler approach for python re.split with capturing group:
            
            # If we want to keep separator attached to the END of chunks (like sentences ending in .)
            # We can use non-consuming split or just simple split and re-attach
            
            splits = re.split(f"({separator_regex})", text)
            # splits will look like [part1, sep1, part2, sep2, ...]
            new_splits = []
            for i in range(1, len(splits), 2):
                new_splits.append(splits[i-1] + splits[i])
            if len(splits) % 2 == 1:
                new_splits.append(splits[-1])
            return [s for s in new_splits if s != ""]
        else:
            return [s for s in re.split(separator_regex, text) if s != ""]

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Combine small splits into chunks of max size with overlap."""
        # If we kept the separator in the splits, we don't need to add it again when joining
        # nor count its length separately
        separator_len = len(separator) if not self._keep_separator else 0
        joiner = separator if not self._keep_separator else ""
        
        docs = []
        current_doc = []
        total = 0
        
        for d in splits:
            _len = len(d)
            
            # If adding this split exceeds chunk size, save current doc and start new one
            if total + _len + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size:
                if total > self._chunk_size:
                    # This implies a single split was larger than chunk size
                    pass
                    
                if current_doc:
                    doc = self._join_docs(current_doc, joiner)
                    if doc is not None:
                        docs.append(doc)
                    
                    # Handle overlap
                    while total > self._chunk_overlap or (total + _len + separator_len > self._chunk_size and total > 0):
                        total -= len(current_doc[0]) + (separator_len if len(current_doc) > 1 else 0)
                        current_doc.pop(0)

            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
            
        doc = self._join_docs(current_doc, joiner)
        if doc is not None:
            docs.append(doc)
            
        return docs

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        text = text.strip()
        return text if text else None


def chunk_transcript(
    text: str,
    chunk_size: int = 1200,  # characters per chunk (approx 300 tokens)
    overlap: int = 200,  # overlap between chunks (approx 50 tokens)
) -> List[Tuple[int, str]]:
    """
    Split transcript into overlapping chunks for embedding using recursive character splitting.
    
    This function uses a hierarchy of separators to split text while preserving 
    semantic structure (paragraphs, sentences) as much as possible.
    
    Default chunk size increased to 1200 chars (~300 tokens) to better utilize 
    embedding model context (512 tokens).

    Args:
        text: The full transcript text
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of (chunk_index, chunk_text) tuples
    """
    if not text or len(text.strip()) == 0:
        return []

    # Clean text: normalize whitespace slightly but preserve paragraph breaks for the splitter
    # We replace 3+ newlines with 2 to keep paragraph structure clean
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    
    # Priority of separators:
    # 1. Double newline (Paragraphs)
    # 2. Single newline (Lines)
    # 3. Sentence endings with space (. ! ?)
    # 4. Words (Space)
    # 5. Characters (fallback)
    separators = ["\n\n", "\n", r"(?<=\. )", r"(?<=\! )", r"(?<=\? )", " ", ""]
    
    splitter = RecursiveCharacterSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators,
        keep_separator=True,
        is_separator_regex=True
    )
    
    chunks_text = splitter.split_text(text)
    
    # Format as tuples (index, text)
    return [(i, chunk) for i, chunk in enumerate(chunks_text)]


def find_sentence_boundary(text: str, start: int, end: int) -> int:
    """
    Deprecated: Kept for backward compatibility if imported elsewhere, 
    but logic moved to RecursiveCharacterSplitter.
    Find the nearest sentence boundary (. ! ?) before end position.
    """
    sentence_endings = [". ", "! ", "? ", ".\n", "!\n", "?\n"]

    # Search backwards from end
    search_text = text[start:end]
    best_pos = end

    for ending in sentence_endings:
        pos = search_text.rfind(ending)
        if pos != -1:
            actual_pos = start + pos + len(ending)
            if actual_pos > start:
                best_pos = min(best_pos, actual_pos)

    return best_pos if best_pos < end else end

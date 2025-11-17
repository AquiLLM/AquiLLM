import re
from dataclasses import dataclass
from typing import List, Optional
from datetime import timedelta

@dataclass
class Caption:
    start_time: timedelta
    end_time: timedelta
    text: str
    speaker: Optional[str] = None
    
    
    def merge_with(self, other: 'Caption') -> 'Caption':
        """Merge this caption with another one"""
        return Caption(
            start_time=self.start_time,
            end_time=other.end_time,
            text=f"{self.text} {other.text}",
            speaker=self.speaker
        )
    
    def can_merge_with(self, other: 'Caption', max_gap: float = 20.0, max_size: int = 1024) -> bool:
        """Check if this caption can be merged with another one"""
        if self.speaker != other.speaker or not self.speaker:
            return False
        if len(self.text) + len(other.text) > max_size:
            return False
        max_gap = timedelta(seconds=max_gap)
        this_end = self.end_time
        other_start = self.start_time
        
        return (other_start - this_end) <= max_gap


def parse_timestamp(timestamp: str) -> timedelta:
    """Convert VTT timestamp to timedelta"""
    match = re.match(r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})', timestamp)
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    
    return timedelta(
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        milliseconds=milliseconds
    )

def parse_content(text: str) -> tuple[Optional[str], str]:
    """Separates speaker from content if present"""
    if ':' in text:
        speaker, content = text.split(':', 1)
        return speaker.strip(), content.strip()
    return None, text.strip()

def parse(file) -> List[Caption]:
    captions = []

    lines = [line.decode('UTF-8').strip() for line in file.readlines()]
    # Skip WEBVTT header
    if lines[0] != 'WEBVTT':
        raise ValueError("File must start with WEBVTT")
    
    i = 1
    while i < len(lines):
        line = lines[i]
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Parse index
        if line.isdigit():
            _ = int(line)
            
            # Parse timestamp line
            i += 1
            if i >= len(lines):
                break
            
            timestamp_line = lines[i]
            try:
                start_time, end_time = timestamp_line.split(' --> ')
                start_time = parse_timestamp(start_time)
                end_time = parse_timestamp(end_time)
            except Exception as e:
                raise ValueError(f"Invalid timestamp line: {timestamp_line}") from e
            
            # Parse content
            i += 1
            if i >= len(lines):
                break
            
            content_line = lines[i]
            speaker, text = parse_content(content_line)
            
            caption = Caption(
                start_time=start_time,
                end_time=end_time,
                text=text,
                speaker=speaker
            )
            captions.append(caption)
        
        i += 1
        
    return captions

def coalesce_captions(captions: List[Caption], max_gap: float = 20.0, max_size: int = 1024) -> List[Caption]:
    """
    Coalesce adjacent captions from the same speaker if they're within max_gap seconds.
    
    Args:
        captions: List of Caption objects
        max_gap: Maximum gap in seconds between captions to consider them for merging
        
    Returns:
        List of coalesced Caption objects
    """
    if not captions:
        return []
        
    coalesced = []
    current = captions[0]
    
    for next_caption in captions[1:]:
        if current.can_merge_with(next_caption, max_gap, max_size):
            current = current.merge_with(next_caption)
        else:
            coalesced.append(current)
            current = next_caption
            
    coalesced.append(current)
    

    return coalesced


# This is not currently being used for chunking because it has no overlap. 
def chunk(captions: List[Caption], chunk_size: int) -> List[List[Caption]]:
    """
    Combine captions into lists of captions of approximately the right chunk size. 
    Chunk size is not required to be less than the exact max chunk size, depending on if I decide to reimplement this later 
    """
    
    if not captions:
        return []
    
    # making each element in captions a list of captions containing only one member
    captions_nested = [[c,] for c in captions]
    
    chunked = []
    current = captions_nested[0]

    for next_caption in captions_nested[1:]:
        total_length = sum(len(c.text) for c in current + next_caption) 
        if  total_length < chunk_size:
            current = current + next_caption 
        else:
            chunked.append(current)
            current = next_caption
    chunked.append(current)

    return chunked
    
def to_text(captions: List[Caption]) -> str:
    ret = ""
    for caption in captions:
        total_seconds = caption.start_time.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        ret += '{:02}:{:02}:{:02} '.format(hours, minutes, seconds) + f'{caption.speaker}: {caption.text}\n\n'
    return ret


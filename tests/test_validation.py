import pytest
from generation.story_generator import StoryGenerator

def test_empty_prompt_rejected_without_loading_model():
    model = StoryGenerator.__new__(StoryGenerator)
    with pytest.raises(ValueError, match="prompt"):
        model.generate(" ")

def test_long_prompt_rejected_without_loading_model():
    model = StoryGenerator.__new__(StoryGenerator)
    with pytest.raises(ValueError, match="1,800"):
        model.generate("x" * 1801)

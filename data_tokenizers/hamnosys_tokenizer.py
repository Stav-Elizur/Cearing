from pathlib import Path
from typing import List
from fontTools.ttLib import TTFont
from data_tokenizers.base_tokenizer import BaseTokenizer


class HamNoSysTokenizer(BaseTokenizer):

    def __init__(self, starting_index=None, **kwargs):
        self.font_path = Path(__file__).parent.joinpath("HamNoSysUnicode.ttf")

        with TTFont(self.font_path) as font:
            tokens = [chr(key) for key in font["cmap"].getBestCmap().keys()]

        super().__init__(tokens=tokens, starting_index=starting_index, **kwargs)

    def text_to_tokens(self, text: str) -> List[str]:
        return [self.bos_token] + list(text)

    def tokens_to_text(self, tokens: List[str]) -> str:
        if tokens[0] == self.bos_token:
            tokens = tokens[1:]

        return "".join(tokens)

import re
import string
from typing import Optional, Tuple


class PromptASTNode:
    """Abstract Syntax Tree node for a parsed prompt."""

    pass


class TextNode(PromptASTNode):
    """Represents a plain text segment."""

    def __init__(self, text: str):
        self.text = text

    def __repr__(self):
        return f"Text('{self.text}')"


class WeightedNode(PromptASTNode):
    """Represents a weighted group of other nodes."""

    def __init__(self, children: list[PromptASTNode], weight: float):
        self.children = children
        self.weight = weight

    def __repr__(self):
        return f"Weighted({self.children}, {self.weight:.2f})"


class RootNode(PromptASTNode):
    """The root of the parsed prompt tree."""

    def __init__(self, children: list[PromptASTNode]):
        self.children = children

    def __repr__(self):
        return f"Root({self.children})"


class NestedPromptParser:
    """
    A recursive descent parser for prompts with nested weights.
    Supports syntax like: "a (blue (sky)++ and green grass)- landscape"
    and bare weighted words like "a red++ car".
    """

    WEIGHT_INCREMENT_FACTOR = 1.6
    """The factor by which weight is multiplied for each '+' or divided for each '-'."""

    def __init__(self, prompt: str):
        self.prompt = prompt
        self.pos = 0
        # Regex for bare weighted words, used on text segments
        # Bare words only support `++` or `--` style weights, not numerical weights.
        # Numerical weights are only supported for parenthesized groups.
        self.bare_word_pattern = re.compile(r"\b([a-zA-Z_]\w*)\b([\+\-]+)(?=[\s,.;:!?)]|$)")
        # self.weight_pattern = re.compile(r"([\+\-]+|[-+]?\d*\.?\d+)(?=[\s,.;:!?)]|$)")
        self.weight_pattern = re.compile(r"([\+\-]+|[-+]?\d*\.?\d+)")

    def _calculate_weight(self, modifier: str) -> float:
        """Calculates the weight from a modifier string like '++', '-', or '1.2'."""
        if not modifier:
            return 1.0
        if modifier.startswith("+"):
            return self.WEIGHT_INCREMENT_FACTOR ** len(modifier)
        elif modifier.startswith("-"):
            return (1 / self.WEIGHT_INCREMENT_FACTOR) ** len(modifier)
        try:
            return float(modifier)
        except ValueError:
            return 1.0

    def _consume_weight(self) -> str:
        """Checks for and consumes a weight modifier at the current position."""
        match = self.weight_pattern.match(self.prompt, self.pos)
        if match:
            weight_str = match.group(1)
            self.pos += len(weight_str)
            return weight_str
        return ""

    def _process_text_segment(self, text: str) -> list[PromptASTNode]:
        """
        Parses a text segment for bare weighted words, e.g. "a red++ car".
        This allows for weighting individual words without parentheses.
        """
        nodes = []
        last_idx = 0
        for match in self.bare_word_pattern.finditer(text):
            # Add the text before the match as a TextNode
            if match.start() > last_idx:
                nodes.append(TextNode(text[last_idx : match.start()]))

            word = match.group(1)
            modifier = match.group(2)
            weight = self._calculate_weight(modifier)

            # A bare weighted word is a WeightedNode containing a single TextNode
            nodes.append(WeightedNode([TextNode(word)], weight))
            last_idx = match.end()

        # Add any remaining text after the last match
        if last_idx < len(text):
            nodes.append(TextNode(text[last_idx:]))

        return nodes

    def _parse_recursive(self, terminator: Optional[str]) -> list[PromptASTNode]:
        """
        Recursively parses segments of the prompt.

        This method handles nested parentheses and escaped characters. It accumulates
        text into a buffer and processes it when a special character (like '(' or a
        terminator) is encountered.
        """
        nodes: list[PromptASTNode] = []
        current_text = ""
        while self.pos < len(self.prompt):
            char = self.prompt[self.pos]

            if terminator and char == terminator:
                if current_text:
                    nodes.extend(self._process_text_segment(current_text))
                return nodes

            if char == "(":
                # Process any accumulated text before handling the parenthesis
                if current_text:
                    nodes.extend(self._process_text_segment(current_text))
                    current_text = ""

                self.pos += 1  # Consume '('
                children = self._parse_recursive(terminator=")")
                self.pos += 1  # Consume ')'

                weight_modifier = self._consume_weight()
                weight = self._calculate_weight(weight_modifier)
                nodes.append(WeightedNode(children, weight))

            elif char == "\\":
                # Handle escaped characters
                self.pos += 1
                if self.pos < len(self.prompt):
                    current_text += self.prompt[self.pos]
                self.pos += 1

            else:
                current_text += char
                self.pos += 1

        if current_text:
            nodes.extend(self._process_text_segment(current_text))

        return nodes

    def parse(self) -> RootNode:
        """Parses the entire prompt string into an AST."""
        self.pos = 0
        children = self._parse_recursive(terminator=None)
        return RootNode(children)


SEPARATORS = string.whitespace + ",.;:!?"


def flatten_ast(node: PromptASTNode, cumulative_weight: float = 1.0) -> list[Tuple[str, float]]:
    """
    Traverses the AST and flattens it into a list of (text, weight) tuples.

    This function applies weights cumulatively down the tree. It also ensures
    proper spacing for tokenization by adding a space between adjacent segments
    that are not already separated by whitespace or punctuation. For example,
    "a(b)" becomes "a (b)" to ensure "a" and "b" are tokenized separately.
    """
    flat_list = []
    if isinstance(node, RootNode) or isinstance(node, WeightedNode):
        if isinstance(node, WeightedNode):
            cumulative_weight *= node.weight

        for i, child in enumerate(node.children):
            child_segments = flatten_ast(child, cumulative_weight)

            if i > 0 and flat_list and child_segments:
                last_text = flat_list[-1][0]
                next_text = child_segments[0][0]
                # Add a space if the previous segment doesn't end with a separator
                # and the next segment doesn't start with one. This is to ensure
                # words that were syntactically separated but not by a space
                # (e.g. `word(word)`) are tokenized separately.
                if last_text and next_text and last_text[-1] not in SEPARATORS and next_text[0] not in SEPARATORS:
                    first_text, first_weight = child_segments[0]
                    child_segments[0] = (" " + first_text, first_weight)

            flat_list.extend(child_segments)
    elif isinstance(node, TextNode):
        # Only add non-empty text
        if node.text:
            flat_list.append((node.text, round(cumulative_weight, 15)))

    return flat_list

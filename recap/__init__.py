"""
RECAP: Reaction Completion for Biochemical Pathways
====================================================

A deep learning framework for predicting missing components in biochemical reactions.

Example usage:
    >>> from recap import ReactionCompleter
    >>> completer = ReactionCompleter.from_pretrained("recap-base")
    >>> result = completer.complete(substrates=["ATP", "glucose"], mode="forward")
    >>> print(result.products)
"""

__version__ = "0.1.0"
__author__ = "Somtirtha Santra"
__email__ = "somtirtha@psu.edu"

from recap.models.completer import ReactionCompleter
from recap.models.reaction_t5 import ReactionT5Model
from recap.data.dataset import ReactionDataset
from recap.data.tokenizer import ReactionTokenizer

__all__ = [
    "ReactionCompleter",
    "ReactionT5Model",
    "ReactionDataset",
    "ReactionTokenizer",
    "__version__",
]

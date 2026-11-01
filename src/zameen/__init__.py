"""zameen: teach a language model to move Urdu poetry between Nastaliq, Devanagari and Roman.

The package turns Rekhta's three-script corpus into instruction data, fine-tunes a model with
LoRA, and evaluates it before and after on poems and poets it never saw.

Heavy dependencies (torch, transformers, peft) are imported only by the GPU backend, so data
preparation, metrics, reporting and the test suite run on any machine.
"""

__version__ = "0.2.0"

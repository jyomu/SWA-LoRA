from dataclasses import dataclass


@dataclass
class AttentionPolicy:
    """Per-layer attention type assignment (sliding_attention / full_attention)."""

    layer_types: list[str]
    sliding_window: int

    @classmethod
    def top_full_layers(
        cls, num_hidden_layers: int, num_full_top_layers: int, sliding_window: int
    ) -> "AttentionPolicy":
        """Lower layers use SWA, the top `num_full_top_layers` layers stay full attention."""
        if not 0 <= num_full_top_layers <= num_hidden_layers:
            raise ValueError("num_full_top_layers must be within [0, num_hidden_layers]")
        layer_types = ["sliding_attention"] * (num_hidden_layers - num_full_top_layers) + [
            "full_attention"
        ] * num_full_top_layers
        return cls(layer_types=layer_types, sliding_window=sliding_window)

    @property
    def swa_layer_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "sliding_attention"]

    @property
    def full_layer_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

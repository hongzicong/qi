from qi.quantization import WeightActivationQuantConfig
from qi.quantization.backends import Int8W8A8Backend, get_backend


def test_int8_w8a8_config_allows_per_token_per_channel() -> None:
    cfg = WeightActivationQuantConfig(
        enabled=True,
        algo="smoothquant",
        quant_dtype="int",
        backend="int8_w8a8",
        activation_granularity="per_token",
        weight_granularity="per_channel",
    )

    cfg.validate()


def test_int8_w8a8_backend_registered() -> None:
    backend = get_backend("int8_w8a8")

    assert isinstance(backend, Int8W8A8Backend)

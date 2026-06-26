import torch

from swa_lora.lora_setup import apply_lora
from swa_lora.toy import build_toy_setup
from swa_lora.trainer import Trainer, TrainerConfig

NUM_LAYERS = 4
NUM_FULL_TOP = 1
WINDOW = 2


def _make_student(rank=4, alpha=8, seed=0):
    setup = build_toy_setup(
        num_hidden_layers=NUM_LAYERS,
        num_full_top_layers=NUM_FULL_TOP,
        sliding_window=WINDOW,
        seed=seed,
    )
    student = apply_lora(setup.student, setup.adapter, setup.policy, rank=rank, alpha=alpha)
    return setup, student


def test_teacher_has_no_gradients_after_backward():
    setup, student = _make_student()
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-2)
    trainer = Trainer(setup.teacher, student, setup.adapter, opt, TrainerConfig())

    input_ids = torch.randint(0, setup.config.vocab_size, (2, 12))
    trainer.train_step(input_ids=input_ids)

    for p in setup.teacher.parameters():
        assert p.grad is None


def test_frozen_full_top_layer_has_no_gradient():
    setup, student = _make_student()
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-2)
    trainer = Trainer(setup.teacher, student, setup.adapter, opt, TrainerConfig())

    input_ids = torch.randint(0, setup.config.vocab_size, (2, 12))
    trainer.train_step(input_ids=input_ids)

    full_layer_idx = setup.policy.full_layer_indices[0]
    full_layer = setup.adapter.decoder_layers(student)[full_layer_idx]
    for p in full_layer.parameters():
        assert p.requires_grad is False
        assert p.grad is None

    final_norm = setup.adapter.final_norm(student)
    lm_head = setup.adapter.lm_head(student)
    embed = setup.adapter.embed_tokens(student)
    for module in (final_norm, lm_head, embed):
        for p in module.parameters():
            assert p.requires_grad is False
            assert p.grad is None


def test_swa_layer_lora_receives_nonzero_gradient():
    setup, student = _make_student()
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-2)
    # grad_accum_steps=2 so train_step stops after backward(), before optimizer.step()
    # zeroes the gradients out -- we want to inspect the raw accumulated grads.
    trainer = Trainer(setup.teacher, student, setup.adapter, opt, TrainerConfig(grad_accum_steps=2))

    input_ids = torch.randint(0, setup.config.vocab_size, (2, 12))
    metrics = trainer.train_step(input_ids=input_ids)
    assert metrics["stepped"] is False

    for layer_idx in setup.policy.swa_layer_indices:
        layer = setup.adapter.decoder_layers(student)[layer_idx]
        attn = setup.adapter.self_attention(layer)
        found_nonzero = False
        for name, p in attn.named_parameters():
            if "lora_" in name and p.requires_grad:
                assert p.grad is not None, f"missing grad for {name}"
                if torch.any(p.grad != 0):
                    found_nonzero = True
        assert found_nonzero, f"layer {layer_idx} LoRA params got no nonzero gradient"


def test_hidden_loss_decreases_when_overfitting_one_batch():
    setup, student = _make_student(rank=8, alpha=16)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=5e-3)
    trainer = Trainer(setup.teacher, student, setup.adapter, opt, TrainerConfig())

    torch.manual_seed(123)
    input_ids = torch.randint(0, setup.config.vocab_size, (2, 16))

    losses = [trainer.train_step(input_ids=input_ids)["loss"].item() for _ in range(60)]

    assert losses[-1] < losses[0] * 0.5


def test_causal_mask_has_no_future_leakage():
    setup, _ = _make_student()
    body = setup.adapter.base_model(setup.student)

    torch.manual_seed(7)
    input_ids = torch.randint(0, setup.config.vocab_size, (1, 10))
    out = body(input_ids=input_ids, output_attentions=True, use_cache=False)

    for layer_attn in out.attentions:
        upper_triangle = torch.triu(layer_attn[0], diagonal=1)
        assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle))


def test_swa_window_attention_is_zero_outside_window():
    setup, _ = _make_student()
    body = setup.adapter.base_model(setup.student)

    torch.manual_seed(7)
    seq_len = 10
    input_ids = torch.randint(0, setup.config.vocab_size, (1, seq_len))
    out = body(input_ids=input_ids, output_attentions=True, use_cache=False)

    for layer_idx in setup.policy.swa_layer_indices:
        attn = out.attentions[layer_idx][0]  # [H, T, T]
        for i in range(seq_len):
            in_window_start = max(0, i - WINDOW + 1)
            outside = attn[:, i, :in_window_start]
            assert torch.allclose(outside, torch.zeros_like(outside))
            in_window = attn[:, i, in_window_start : i + 1]
            assert torch.all(in_window.sum(dim=-1) > 0.999)


def test_full_top_layer_attends_to_all_past_positions():
    setup, _ = _make_student()
    body = setup.adapter.base_model(setup.student)

    torch.manual_seed(7)
    seq_len = 10
    input_ids = torch.randint(0, setup.config.vocab_size, (1, seq_len))
    out = body(input_ids=input_ids, output_attentions=True, use_cache=False)

    full_layer_idx = setup.policy.full_layer_indices[0]
    attn = out.attentions[full_layer_idx][0]  # [H, T, T]
    last_query_row = attn[:, -1, :]
    assert torch.all(last_query_row > 0), "full attention layer should attend to every past position"


def test_set_sliding_window_changes_window_mid_run():
    setup, _ = _make_student()
    body = setup.adapter.base_model(setup.student)

    torch.manual_seed(7)
    seq_len = 10
    input_ids = torch.randint(0, setup.config.vocab_size, (1, seq_len))

    swa_layer_idx = setup.policy.swa_layer_indices[0]
    row = 5

    out_before = body(input_ids=input_ids, output_attentions=True, use_cache=False)
    attn_before = out_before.attentions[swa_layer_idx][0, 0, row]
    assert int((attn_before > 0).sum()) == WINDOW

    wider_window = WINDOW + 3
    setup.adapter.set_sliding_window(setup.student, wider_window)

    out_after = body(input_ids=input_ids, output_attentions=True, use_cache=False)
    attn_after = out_after.attentions[swa_layer_idx][0, 0, row]
    assert int((attn_after > 0).sum()) == wider_window


def test_layerwise_attention_loss_only_trains_its_own_layer():
    setup, student = _make_student()

    torch.manual_seed(11)
    input_ids = torch.randint(0, setup.config.vocab_size, (2, 12))

    swa_indices = setup.policy.swa_layer_indices
    assert len(swa_indices) >= 2, "need at least 2 SWA layers to test cross-layer leakage"
    target_layer = swa_indices[0]
    other_layers = swa_indices[1:]

    loss, per_layer = setup.adapter.layerwise_attention_loss(
        setup.teacher, student, [target_layer], input_ids, lambda_cos=1.0
    )
    assert set(per_layer.keys()) == {target_layer}
    loss.backward()

    target_attn = setup.adapter.self_attention(setup.adapter.decoder_layers(student)[target_layer])
    found_nonzero = False
    for name, p in target_attn.named_parameters():
        if "lora_" in name and p.requires_grad and p.grad is not None and torch.any(p.grad != 0):
            found_nonzero = True
    assert found_nonzero, "targeted layer's LoRA should receive nonzero gradient"

    for layer_idx in other_layers:
        other_attn = setup.adapter.self_attention(setup.adapter.decoder_layers(student)[layer_idx])
        for name, p in other_attn.named_parameters():
            if "lora_" in name:
                assert p.grad is None, f"layer {layer_idx} ({name}) should not receive gradient from layer {target_layer}'s loss"

    for p in setup.teacher.parameters():
        assert p.grad is None

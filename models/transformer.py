import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass
import optax


@dataclass
class TransformerConfig:
    vocab_size: int
    emb_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    mlp_dim: int = 512
    max_len: int = 512
    dropout_rate: float = 0.1


class PositionalEncoding(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x):
        seq_len = x.shape[1]
        position_ids = jnp.arange(seq_len)
        pos_emb = nn.Embed(
            num_embeddings=self.config.max_len, features=self.config.emb_dim
        )(position_ids)
        pos_emb = pos_emb[None, :, :]
        return pos_emb


class TransformerBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, mask=None, train=True):
        attn = nn.SelfAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.emb_dim,
            dropout_rate=self.config.dropout_rate,
            use_bias=True,
        )
        y = attn(x, mask=mask, deterministic=not train)
        x = x + y
        x = nn.LayerNorm()(x)

        y = nn.Dense(self.config.mlp_dim)(x)
        y = nn.gelu(y)
        y = nn.Dense(self.config.emb_dim)(y)
        x = x + y
        x = nn.LayerNorm()(x)
        return x


class DecoderOnlyTransformer(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, input_ids, padding_mask, train=True):
        x = nn.Embed(
            num_embeddings=self.config.vocab_size, features=self.config.emb_dim
        )(input_ids)
        x = x + PositionalEncoding(self.config)(x)

        padding_mask = nn.make_attention_mask(padding_mask, padding_mask, dtype=jnp.float32)
        causal_mask = nn.make_causal_mask(input_ids, dtype=jnp.float32)
        mask = nn.combine_masks(causal_mask, padding_mask)

        for _ in range(self.config.num_layers):
            x = TransformerBlock(self.config)(x, mask=mask, train=train)

        logits = nn.Dense(self.config.vocab_size)(x)
        return logits

@jax.jit
def train_step(state, batch, dropout_rng):
    input_ids = batch["input_ids"]
    padding_mask = batch["padding_mask"]

    src = input_ids[:, :-1]
    tgt = input_ids[:, 1:]
    tgt_padding_mask = padding_mask[:, 1:]

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, src, tgt_padding_mask, train=True, rngs={'dropout': dropout_rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=tgt)
        return loss.mean()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, batch):
    input_ids = batch["input_ids"]
    padding_mask = batch["padding_mask"]

    src = input_ids[:, :-1]
    tgt = input_ids[:, 1:]
    tgt_padding_mask = padding_mask[:, 1:]

    logits = state.apply_fn(
        {'params': state.params},
        src, tgt_padding_mask,
        train=False,
        rngs={'dropout': jax.random.PRNGKey(0)}
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=tgt).mean()

    output_ids = jnp.argmax(logits, axis=-1)
    return loss, output_ids

if __name__ == "__main__":
    from utils.tokenizer import MultiplicationTokenizer
    from data.multiplication import MultiplicationDataModule
    # from pprint import pprint as print

    tokenizer = MultiplicationTokenizer()
    rng = jax.random.PRNGKey(0)
    data_module = MultiplicationDataModule(
        num_digits=2, num_samples=1000, batch_size=1, tokenizer=tokenizer, max_length=128, rng=rng
    )

    train_loader = data_module.get_train_dataloader()

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        emb_dim=16,
        num_heads=2,
        num_layers=1,
        mlp_dim=64,
        max_len=128,
    )
    model = DecoderOnlyTransformer(config)

    rng = jax.random.PRNGKey(0)
    input_ids = jnp.empty((2, 10), dtype=jnp.int32)
    padding_mask = jnp.ones((2, 10), dtype=jnp.float32)

    params_rng, dropout_rng = jax.random.split(rng)
    params = model.init(rng, input_ids, padding_mask)
    # print(params)

    print(jax.tree_util.tree_map(lambda x: x.shape, params))

    for batch in train_loader:
        input_ids = batch["input_ids"]
        padding_mask = batch["padding_mask"]

        logits = model.apply(
            params, input_ids, padding_mask, train=True, rngs={"dropout": dropout_rng}
        )

        print(logits)
        print(logits.shape)
        break

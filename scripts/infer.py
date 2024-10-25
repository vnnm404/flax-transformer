import jax
from jax import numpy as jnp
import jax.random
import numpy as np
from flax.training import train_state
from flax import struct
import optax
from tqdm import tqdm
import orbax.checkpoint
from flax.training import orbax_utils

from utils.tokenizer import MultiplicationTokenizer
from data.multiplication import MultiplicationDataModule
from models.transformer import DecoderOnlyTransformer, TransformerConfig, train_step, eval_step

def main(input="b35x65="):
    tokenizer = MultiplicationTokenizer()

    rng = jax.random.PRNGKey(0)
    data_module = MultiplicationDataModule(
        num_digits=2, num_samples=1000, batch_size=128, tokenizer=tokenizer, max_length=128, rng=rng
    )
    
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        emb_dim=128,
        num_heads=8,
        num_layers=6,
        mlp_dim=4 * 128,
        max_len=128,
    )

    model = DecoderOnlyTransformer(config)

    ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    ckpt = ckptr.restore("/home/sreeramv/implicit-cot/cache/dev", item=None)
    params = ckpt["model"]["params"]

    input_ids = tokenizer.encode(input)
    input_ids = jnp.array(np.array(input_ids, dtype=np.int32).reshape(1, -1))
    # print(input_ids)

    for _ in tqdm(range(128 - len(input_ids[0]))):
        padding_mask = jnp.ones_like(input_ids, dtype=jnp.float32)
        logits = model.apply({ "params": params }, input_ids, padding_mask, train=False)
        
        output_id = jnp.argmax(logits[:, -1], axis=-1).reshape(1, 1)
        input_ids = jnp.concatenate([input_ids, output_id], axis=-1)

    decoded_output = tokenizer.decode(np.array(input_ids[0]))
    print(decoded_output)

    # eval_loader = data_module.get_val_dataloader()
    # for batch in eval_loader:
    #     input_ids = batch["input_ids"][:, :-1]
    #     padding_mask = jnp.ones_like(input_ids, dtype=jnp.float32)

    #     logits = model.apply(
    #         { "params": params }, input_ids, padding_mask, train=False
    #     )

    #     output_ids = jnp.argmax(logits, axis=-1)

    #     input_ids = batch["input_ids"]
    #     full_texts = [tokenizer.decode(np.array(input_id)) for input_id in input_ids]

    #     decoded_outputs = [tokenizer.decode(np.array(output)) for output in output_ids]
    #     break

    # print("gt", full_texts[0])
    # print("yy", decoded_outputs[0])

    # print("gt", full_texts[1])
    # print("yy", decoded_outputs[1])

if __name__ == "__main__":
    main()

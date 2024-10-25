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

class TrainState(train_state.TrainState):
    pass

def create_train_state(module, rng, learning_rate):
    input_ids = jnp.empty((1, 10), dtype=jnp.int32)
    padding_mask = jnp.ones((1, 10), dtype=jnp.float32)
    params = module.init(rng, input_ids, padding_mask)["params"]
    
    tx = optax.adam(learning_rate=learning_rate)
    
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx
    )

def exact_match(outputs, full_texts):
    total_correct = 0

    for output, full_text in zip(outputs, full_texts):
        pred = output.split("=")[-1].replace("e", "")
        gt = full_text.split("=")[-1].replace("e", "")

        # print(pred, gt)

        if pred == gt:
            total_correct += 1
    
    return total_correct, len(full_texts)

def main():
    tokenizer = MultiplicationTokenizer()

    rng = jax.random.PRNGKey(0)
    data_module = MultiplicationDataModule(
        num_digits=2, num_samples=1000, batch_size=128, tokenizer=tokenizer, max_length=128, rng=rng
    )

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        emb_dim=256,
        num_heads=8,
        num_layers=6,
        mlp_dim=4 * 256,
        max_len=128,
    )

    model = DecoderOnlyTransformer(config)
    # print(
    #     model.tabulate(
    #         rng,  # same key since we are not using randomness
    #         jnp.empty((1, 16), dtype=jnp.int32),
    #         # compute_flops=True,
    #         # compute_vjp_flops=True,
    #     )
    # )

    init_rng = jax.random.PRNGKey(1)
    state = create_train_state(model, init_rng, learning_rate=1e-4)
    del init_rng

    # train loop
    num_epochs = 1000
    key = jax.random.PRNGKey(2)
    lowest_val_loss = 1e9
    
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        dropout_rng, key = jax.random.split(key)

        # train
        train_total_loss = 0
        train_loader = data_module.get_train_dataloader()
        
        for batch in train_loader:
            state, loss = train_step(state, batch, dropout_rng)
            train_total_loss += loss
        
        train_avg_loss = train_total_loss / 49

        # eval
        val_total_loss = 0
        eval_loader = data_module.get_val_dataloader()

        total_correct, total_samples = 0, 0
        for batch in eval_loader:
            loss, output_ids = eval_step(state, batch)
            val_total_loss += loss

            # decode the input_ids
            input_ids = batch["input_ids"]
            full_texts = [tokenizer.decode(np.array(input_id)) for input_id in input_ids]

            # decode the output_ids
            decoded_outputs = [tokenizer.decode(np.array(output)) for output in output_ids]
            correct, samples = exact_match(decoded_outputs, full_texts)

            total_correct += correct
            total_samples += samples
        
        # print("gt", full_texts[0])
        # print("yy", decoded_outputs[0])
        
        val_avg_loss = val_total_loss / 49
        exact_match_accuracy = total_correct / total_samples
        
        pbar.set_postfix(
            {
                'train_loss': '{:.3f}'.format(train_avg_loss), 
                'val_loss': '{:.3f}'.format(val_avg_loss),
                'exact_match_accuracy': '{:.2f}'.format(exact_match_accuracy)
            }
        )

        if val_avg_loss < lowest_val_loss:
            lowest_val_loss = val_avg_loss

            ckpt = {"model": state, "lowest_val_loss": lowest_val_loss}
            save_args = orbax_utils.save_args_from_target(ckpt)

            ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
            ckptr.save("/home/sreeramv/implicit-cot/cache/dev", ckpt, save_args=save_args, force=True)

if __name__ == "__main__":
    main()

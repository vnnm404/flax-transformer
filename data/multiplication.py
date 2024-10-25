import itertools
import numpy as np
import jax


class MultiplicationDataModule:
    def __init__(self, num_digits, num_samples, batch_size, tokenizer, max_length, rng):
        self.num_digits = num_digits
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prng_key = rng
        self.setup()

    def setup(self):
        self.prng_key, train_key, val_key, test_key = jax.random.split(self.prng_key, 4)
        num_train_samples = int(self.num_samples * 0.8)
        num_val_samples = int(self.num_samples * 0.1)
        num_test_samples = self.num_samples - num_train_samples - num_val_samples
        self.train_dataset = self.generate_dataset(
            train_key, num_train_samples, test=False
        )
        self.val_dataset = self.generate_dataset(val_key, num_val_samples, test=False)
        self.test_dataset = self.generate_dataset(test_key, num_test_samples, test=True)

    def generate_dataset(self, prng_key, num_samples, test=False):
        subkeys = jax.random.split(prng_key, 2)
        key_a, key_b = subkeys
        a_values = jax.random.randint(
            key_a, (num_samples,), 10 ** (self.num_digits - 1), 10**self.num_digits
        )
        b_values = jax.random.randint(
            key_b, (num_samples,), 10 ** (self.num_digits - 1), 10**self.num_digits
        )
        dataset = []
        for a, b in zip(a_values, b_values):
            example = self.create_example(int(a), int(b), test)
            dataset.append(example)
        return dataset

    def create_example(self, a, b, test=False):
        result = a * b
        equation = f"{a}x{b}="
        expanded = self.expand_multiplication(a, b)
        solution = f"{result}"
        
        if test:
            full_text = f"{self.tokenizer.bos_token}{equation}"
        else:
            full_text = f"{self.tokenizer.bos_token}{equation}{expanded}{solution}{self.tokenizer.eos_token}"
            # full_text = f"{self.tokenizer.bos_token}{equation}"

        encoded = self.tokenizer.encode(full_text, max_length=self.max_length)
        return {
            "input_ids": np.array(encoded, dtype=np.int32),
            "full_text": full_text,
        }

    def expand_multiplication(self, a, b):
        a_digits = [int(d) for d in str(a)]
        b_digits = [int(d) for d in str(b)]
        expanded = (
            "("
            + "+".join([f"{d}x{10 ** i}" for i, d in enumerate(reversed(a_digits))])
            + ")x("
        )
        expanded += (
            "+".join([f"{d}x{10 ** i}" for i, d in enumerate(reversed(b_digits))])
            + ")="
        )
        terms = [
            f"{da}x{db}x{10 ** i}x{10 ** j}"
            for (i, da), (j, db) in itertools.product(
                enumerate(reversed(a_digits)), enumerate(reversed(b_digits))
            )
        ]
        expanded += "+".join(terms) + "="
        partial_sums = [
            str(da * db * (10**i) * (10**j))
            for (i, da), (j, db) in itertools.product(
                enumerate(reversed(a_digits)), enumerate(reversed(b_digits))
            )
        ]
        expanded += "+".join(partial_sums) + "="
        return expanded

    # def pad_sequences(self, sequences):
    #     max_length = max(len(seq) for seq in sequences)
    #     batch_size = len(sequences)
    #     pad_token_id = self.tokenizer.pad_token_id
    #     padded_sequences = np.full(
    #         (batch_size, max_length), fill_value=pad_token_id, dtype=np.int32
    #     )
    #     for i, seq in enumerate(sequences):
    #         padded_sequences[i, : len(seq)] = seq
    #     return padded_sequences

    def collate_batch(self, batch):
        input_ids = np.stack([item["input_ids"] for item in batch])
        # full_texts = [item["full_text"] for item in batch]

        pad_token_id = self.tokenizer.pad_token_id
        padding_mask = (input_ids != pad_token_id).astype(np.float32)

        return {
            "input_ids": input_ids,
            "padding_mask": padding_mask,
        }

    def get_dataloader(self, dataset, shuffle=True, prng_key=None):
        indices = np.arange(len(dataset))
        if shuffle:
            if prng_key is None:
                raise ValueError("prng_key must be provided when shuffle is True")
            indices = np.array(jax.random.permutation(prng_key, len(dataset)))
        for start_idx in range(0, len(dataset), self.batch_size):
            batch_indices = indices[start_idx : start_idx + self.batch_size]
            batch = [dataset[i] for i in batch_indices]
            yield self.collate_batch(batch)

    def get_train_dataloader(self):
        self.prng_key, subkey = jax.random.split(self.prng_key)
        return self.get_dataloader(self.train_dataset, shuffle=True, prng_key=subkey)

    def get_val_dataloader(self):
        return self.get_dataloader(self.val_dataset, shuffle=False)

    def get_test_dataloader(self):
        return self.get_dataloader(self.test_dataset, shuffle=False)


if __name__ == "__main__":
    from utils.tokenizer import MultiplicationTokenizer
    from pprint import pprint

    tokenizer = MultiplicationTokenizer()
    rng = jax.random.PRNGKey(0)  # Seed for reproducibility
    data_module = MultiplicationDataModule(
        num_digits=2, num_samples=1000, batch_size=2, tokenizer=tokenizer, max_length=128, rng=rng
    )

    train_loader = data_module.get_train_dataloader()
    for batch in train_loader:
        pprint(batch['input_ids'])

        decoded = [tokenizer.decode(ids) for ids in batch['input_ids']]
        pprint(decoded)
        break

    test_loader = data_module.get_test_dataloader()
    for batch in test_loader:
        # pprint(batch["solutions"])
        pprint(batch['input_ids'])

        decoded = [tokenizer.decode(ids) for ids in batch['input_ids']]
        pprint(decoded)
        break

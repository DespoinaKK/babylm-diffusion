import torch
import random
import torch.nn.functional as F

class SpanMaskingStrategy:
    def __init__(self, n_special_tokens, random_p, keep_p, vocab_size, mask_token_id):
        self.n_special_tokens = n_special_tokens
        self.random_p = random_p
        self.keep_p = keep_p
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.max_span_length = 3

    def __call__(self, tokens):
        length = tokens.size(0)

        span_lengths = torch.randint(1, self.max_span_length + 1, size=(length,), dtype=torch.int)
        cumsum = torch.cumsum(span_lengths, dim=0)

        total_length = cumsum[-1].item()
        indices = torch.zeros(total_length, dtype=torch.int)
        indices[cumsum - span_lengths] = torch.arange(length, dtype=torch.int)
        indices = torch.cummax(indices, dim=0)[0]
        indices = indices[:length]

        max_index = indices[-1].item()
        span_random_numbers_1, span_random_numbers_2 = torch.rand([(max_index + 1) * 2]).chunk(2)

        mask_ratios = span_random_numbers_1[indices]

        mask_ratios[tokens < self.n_special_tokens] = float('inf')

        replacement_p = span_random_numbers_2[indices]
        random_mask = replacement_p < self.random_p

        replacement_tokens = tokens.clone()
        replacement_tokens[random_mask] = torch.randint(
            low=self.n_special_tokens,
            high=self.vocab_size,
            size=[random_mask.sum().item()],
            dtype=torch.long
        )
        replacement_tokens[replacement_p > (self.random_p + self.keep_p)] = self.mask_token_id

        return mask_ratios, replacement_tokens

class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, input_file: str, tokenizer, args):
        self.path = input_file
        self.seq_length = args.seq_length
        self.n_special_tokens = args.n_special_tokens

        self.mask_index = tokenizer.token_to_id("<mask>")
        self.cls_index = tokenizer.token_to_id("<s>")
        self.sep_index = tokenizer.token_to_id("</s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        # Check for missing tokens
        if any(token_id is None for token_id in [self.mask_index, self.cls_index, self.sep_index, self.pad_index]):
            raise ValueError("Required tokens not found in tokenizer")

        self.masking_strategy = SpanMaskingStrategy(args.n_special_tokens, args.mask_random_p, args.mask_keep_p, args.vocab_size, self.mask_index)

        try:
            documents = torch.load(input_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load {input_file}: {e}")

        self.segments = [
            document[offset : offset + self.seq_length - 2]
            for document in documents
            for offset in range(0, len(document), self.seq_length - 2)
            if len(document) > 0 and len(document) - offset > 1
        ]
        if hasattr(args, "rank") and args.rank is not None:
            self.segments = self.segments[args.rank::args.world_size]
            random.seed(args.rank)
        else:
            random.seed(getattr(args, 'seed', 42))
        random.shuffle(self.segments)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        tokens = self.segments[index]
        seq_length = min(self.seq_length - 2, tokens.size(0))

        segment = torch.cat([
            torch.LongTensor([self.cls_index]),
            tokens[:seq_length].long()
        ])
        attention_mask = torch.ones(seq_length + 1, seq_length + 1, dtype=torch.bool)

        mask_ratios, replacement_tokens = self.masking_strategy(segment)
        input_ids, target_ids, real_mask_p = self.apply_mask(segment, mask_ratios, replacement_tokens)

        padding_length = self.seq_length - segment.size(0) + 1
        if padding_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.pad_index] * padding_length)
            ])
            target_ids = torch.cat([
                target_ids,
                torch.LongTensor([-100] * padding_length)
            ])
            attention_mask = torch.block_diag(
                attention_mask,
                torch.zeros(padding_length, padding_length, dtype=torch.bool)
            )

        attention_mask = ~attention_mask

        input_ids = input_ids[:-1]
        target_ids = target_ids[:-1]
        attention_mask = attention_mask[:-1, :-1]

        return input_ids, target_ids, attention_mask, real_mask_p

    def apply_mask(self, input_ids, mask_ratios, replacement_ids):
        mask_p = getattr(self, 'validation_mask_p', 0.15)  # Made configurable
        mask_p = torch.topk(mask_ratios, max(1, int(mask_ratios.size(0) * mask_p + 0.5)), largest=False).values.max().item()

        mask = mask_ratios < mask_p
        target_ids = torch.where(mask, input_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)

        real_mask_p = mask.sum().float() / mask_ratios.numel()

        return input_ids, target_ids, real_mask_p.item()


class GenericDataset(torch.utils.data.Dataset):
    def __init__(self, input_file: str, tokenizer, args, seq_length, rank, world_size):
        self.path = input_file
        self.seq_length = seq_length
        self.n_special_tokens = args.n_special_tokens
        self.args = args

        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        # Check for missing tokens
        if self.cls_index is None or self.pad_index is None:
            raise ValueError("Required tokens not found in tokenizer")

        try:
            documents = torch.load(input_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load {input_file}: {e}")
            
        print(f"Loaded {len(documents)} documents")

        # Document-aware segmentation with hard truncation
        self.segments = []
        
        for document in documents:
            if len(document) == 0:
                continue
                
            # Split document into non-overlapping chunks
            for offset in range(0, len(document), self.seq_length - 2):
                segment = document[offset:offset + self.seq_length - 2]
                if len(segment) > 1:  # Only keep non-trivial segments
                    self.segments.append(segment)
        
        print(f"Created {len(self.segments)} segments from {len(documents)} documents")

        if rank is not None:
            self.segments = self.segments[rank::world_size]
        print("Process with rank", rank, "processes ", len(self.segments), "sequences")
                
    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        tokens = self.segments[index]
        seq_length = min(self.seq_length, tokens.size(0))
        tokens = tokens[:seq_length].long()

        input_ids = torch.cat([
            torch.LongTensor([self.cls_index]),
            tokens
        ])
        attention_mask = torch.ones(seq_length + 1, seq_length + 1, dtype=torch.bool)

        padding_length = self.seq_length - input_ids.size(0) + 1
        if padding_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.pad_index] * padding_length)
            ])
            attention_mask = torch.block_diag(
                attention_mask,
                torch.zeros(padding_length, padding_length, dtype=torch.bool)
            )

        attention_mask = ~attention_mask

        input_ids = input_ids[:-1]
        attention_mask = attention_mask[:-1, :-1]

        return input_ids, attention_mask

    def show_random_item(self, tokenizer):
        if len(self) == 0:
            print("Dataset is empty")
            return
        input_ids, attention_mask = self.__getitem__(torch.randint(0, len(self), []).item())
        print(' '.join(tokenizer.id_to_token(i) for i in input_ids.tolist()), flush=True)
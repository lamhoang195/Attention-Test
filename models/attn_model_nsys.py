import torch
from .model import Model
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import sample_token

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AttentionModelNoSys(Model):
    def __init__(self, config):
        super().__init__(config)
        self.name = config["model_info"]["name"]
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        model_id = config["model_info"]["model_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager"
        ).eval()
        if config["params"]["important_heads"] == "all":
            attn_size = self.get_map_dim()
            self.important_heads = [[i, j] for i in range(
                attn_size[0]) for j in range(attn_size[1])]
        else:
            self.important_heads = config["params"]["important_heads"]
        
        self.top_k = 50
        self.top_p = None

    def get_map_dim(self):
        _, _, attention_maps, _, _, _ = self.inference("print hi", "")
        attention_map = attention_maps[0]
        return len(attention_map), attention_map[0].shape[1]

    def inference(self, instruction, data, max_output_tokens=None):
        data = "Data: " + data
        messages = [
            {"role": "user", "content": instruction + "\n" + data},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        instruction_len = len(self.tokenizer.encode(instruction))
        data_len = len(self.tokenizer.encode(data))

        model_inputs = self.tokenizer(
            [text], return_tensors="pt").to(self.model.device)
        input_tokens = self.tokenizer.convert_ids_to_tokens(
            model_inputs['input_ids'][0])

        if "gemma2_9b-attn" in self.name:
            data_range = ((5, 5+instruction_len), (-4-data_len, -5))
        else:
            raise NotImplementedError

        generated_tokens = []
        generated_probs = []
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask

        if max_output_tokens != None:
            n_tokens = max_output_tokens
        else:
            n_tokens = self.max_output_tokens

        with torch.no_grad():
            # Step 1: Chạy bước đầu tiên với attentions
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )

            # Extract logits và generate token đầu tiên
            logits = output.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token_id = sample_token(
                logits[0], top_k=self.top_k, top_p=None, temperature=1.0)[0]

            generated_probs.append(probs[0, next_token_id.item()].item())
            generated_tokens.append(next_token_id.item())

            # Lưu attention của token đầu tiên
            attention_map = [attention.detach().cpu().half()
                             for attention in output['attentions']]
            attention_map = [torch.nan_to_num(
                attention, nan=0.0) for attention in attention_map]
            attention_map = get_last_attn(attention_map)  # Chỉ lấy attention của token cuối
            attention_maps = [attention_map]
            del output  # Giải phóng memory

            if next_token_id.item() == self.tokenizer.eos_token_id:
                output_tokens = [self.tokenizer.decode(
                    token, skip_special_tokens=True) for token in generated_tokens]
                generated_text = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True)
                return generated_text, output_tokens, attention_maps, input_tokens, data_range, generated_probs

            # Update input cho step tiếp theo
            input_ids = torch.cat(
                (input_ids, next_token_id.unsqueeze(0).unsqueeze(0)), dim=-1)
            attention_mask = torch.cat(
                (attention_mask, torch.tensor([[1]], device=input_ids.device)), dim=-1)

            # Step 2+: Chạy các token còn lại KHÔNG có attentions
            remaining = n_tokens - 1
            for _ in range(remaining):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=False  # Tắt attentions để tiết kiệm memory
                )

                logits = output.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token_id = sample_token(
                    logits[0], top_k=self.top_k, top_p=None, temperature=1.0)[0]

                generated_probs.append(probs[0, next_token_id.item()].item())
                generated_tokens.append(next_token_id.item())

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

                input_ids = torch.cat(
                    (input_ids, next_token_id.unsqueeze(0).unsqueeze(0)), dim=-1)
                attention_mask = torch.cat(
                    (attention_mask, torch.tensor([[1]], device=input_ids.device)), dim=-1)

        output_tokens = [self.tokenizer.decode(
            token, skip_special_tokens=True) for token in generated_tokens]
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True)

        return generated_text, output_tokens, attention_maps, input_tokens, data_range, generated_probs
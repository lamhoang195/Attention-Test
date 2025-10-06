import argparse
import random
import torch
import numpy as np
import re
from utils import open_config, create_model
from detector.attn import AttentionDetector

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(args.seed)

    model_config_path = f"./configs/model_configs/{args.model_name}_config.json"
    model_config = open_config(config_path=model_config_path)
    model = create_model(config=model_config)
    model.print_model_info()

    detector = AttentionDetector(model)

    # Tạo file dataset riêng cho model
    payload_output_file = f"./results/{args.model_name}_payload.txt"

    with open(args.input_file, "r", encoding="utf-8") as f:
        idx = 0
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Bỏ số thứ tự kiểu "123 " ở đầu dòng nếu có
            text = re.sub(r'^\s*\d+\s+', '', line)

            if not text:
                continue

            # Reset seed before each payload to keep outputs deterministic across multiple runs
            set_seed(args.seed)

            # Always use the standard detect path for consistency
            result = detector.detect(text)

            idx += 1
            output_one_line = result[1]['generated_text'].replace("\n", " ").strip()

            print(f"=== Sample {idx} ===")
            print(f"Input: {text}")
            print(f"Output: {output_one_line}")
            print(f"Score: {result[1]['focus_score']}")
            print(f"Detected Injection: {result[0]}")
            print("Check: ")
            print()

            # Ghi kết quả từng sample vào file dataset riêng
            with open(payload_output_file, "a", encoding="utf-8") as f:
                f.write(f"=== Sample {idx} ===\n")
                f.write(f"Input: {text}\n")
                f.write(f"Output: {output_one_line}\n")
                f.write(f"Score: {result[1]['focus_score']}\n")
                f.write(f"Detected Injection: {result[0]}\n")
                f.write("Check: ")
                f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run detector on each line of a text file")
    parser.add_argument("--model_name", type=str, default="qwen2-attn")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast", action="store_true", help="Use fast inference mode")
    args = parser.parse_args()
    main(args)
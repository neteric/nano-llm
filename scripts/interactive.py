"""预训练模型交互续写。输入故事开头，模型接着写。

用法:
    python scripts/interactive.py \
        --ckpt checkpoints/tinystories/pretrain_final.pt \
        --tokenizer data/tinystories_tokenizer.json \
        --device cuda --dtype bfloat16
"""
import argparse
import torch
from nanollm.config import ModelConfig
from nanollm.model import NanoLLM
from nanollm.tokenizer import NanoTokenizer
from nanollm.utils import detect_device, get_dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    device = detect_device(args.device)
    dtype = get_dtype(args.dtype)

    tk = NanoTokenizer.load(args.tokenizer)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = NanoLLM(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    print(f"== 模型加载完成 ({model.num_parameters()/1e6:.1f}M params) ==")
    print("输入故事开头，模型接着续写。/exit 退出，/temp <值> 调温度。\n")

    temperature = args.temperature
    while True:
        try:
            prompt = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if not prompt:
            continue
        if prompt == "/exit":
            break
        if prompt.startswith("/temp "):
            try:
                temperature = float(prompt.split()[1])
                print(f"[温度设为 {temperature}]")
            except ValueError:
                print("[用法: /temp 0.8]")
            continue

        ids = tk.encode(prompt, add_bos=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=dtype):
                out = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
        new_tokens = out[0, len(ids):].tolist()
        print(f"模型: {prompt}{tk.decode(new_tokens)}\n")


if __name__ == "__main__":
    main()

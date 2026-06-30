"""推理脚本: 加载 checkpoint 并生成文本。

两种模式:
  1) 续写 (适合预训练模型):
     python scripts/generate.py --ckpt checkpoints/pretrain_final.pt \
            --tokenizer data/tokenizer.json --prompt "从前有座山，山里有个"
  2) 对话 (适合 SFT 后的模型):
     python scripts/generate.py --ckpt checkpoints/sft_final.pt \
            --tokenizer data/tokenizer.json --chat
"""
import argparse
import sys

import torch

from nanollm.config import ModelConfig
from nanollm.model import NanoLLM
from nanollm.tokenizer import NanoTokenizer
from nanollm.utils import detect_device, get_dtype, human_count


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**ckpt["model_config"])
    model = NanoLLM(mcfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, mcfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default=None, help="续写模式的输入")
    ap.add_argument("--chat", action="store_true", help="交互式对话模式")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    device = detect_device(args.device)
    dtype = get_dtype(args.dtype)
    print(f"== device = {device} ==")

    tk = NanoTokenizer.load(args.tokenizer)
    model, mcfg = load_model(args.ckpt, device)
    # 用 autocast 提速；模型权重保持 fp32，bf16 只在前向计算时启用
    print(f"== loaded {args.ckpt} ({human_count(model.num_parameters())} params) ==\n")

    if args.chat:
        history = []
        print("进入对话模式。输入 /reset 清空历史，/exit 退出。\n")
        while True:
            try:
                user = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user:
                continue
            if user == "/exit":
                break
            if user == "/reset":
                history = []
                print("[历史已清空]")
                continue
            history.append({"role": "user", "content": user})
            ids = tk.apply_chat_template(history, add_generation_prompt=True)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.amp.autocast(device_type=device, dtype=dtype) if device != "cpu" else torch.amp.autocast("cpu", enabled=False):
                out = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    eos_token_id=tk.end_id,
                )
            new_tokens = out[0, input_ids.shape[1]:].tolist()
            # 去掉末尾的 <end>
            if new_tokens and new_tokens[-1] == tk.end_id:
                new_tokens = new_tokens[:-1]
            reply = tk.decode(new_tokens)
            history.append({"role": "assistant", "content": reply})
            print(f"模型: {reply}\n")
        return

    # 续写模式
    if args.prompt is None:
        print("请用 --prompt 提供输入，或加 --chat 进入对话模式", file=sys.stderr)
        sys.exit(1)
    ids = tk.encode(args.prompt, add_bos=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    print(f"[prompt] {args.prompt}\n")
    print("[生成] ", end="", flush=True)
    with torch.amp.autocast(device_type=device, dtype=dtype) if device != "cpu" else torch.amp.autocast("cpu", enabled=False):
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=tk.eos_id,
        )
    new_tokens = out[0, input_ids.shape[1]:].tolist()
    print(tk.decode(new_tokens))


if __name__ == "__main__":
    main()

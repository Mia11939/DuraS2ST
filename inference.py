#!/usr/bin/env python3
"""Translate one English or Chinese audio file with DuraS2ST."""
import argparse
import asyncio
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local model directory or Hugging Face model ID.')
    parser.add_argument('--input-audio', type=Path, required=True)
    parser.add_argument('--target-language', choices=['zh', 'en'], required=True)
    parser.add_argument('--output-audio', type=Path, default=Path('outputs/translation.wav'))
    parser.add_argument('--show-reasoning', action='store_true')
    return parser.parse_args()


async def run(args):
    audio = args.input_audio.expanduser().resolve()
    output = args.output_audio.expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(audio)
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a different output filename')
    model_path = Path(args.model).expanduser()
    if not model_path.is_dir():
        if model_path.is_absolute() or args.model.startswith('.'):
            raise FileNotFoundError(model_path)
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(args.model))
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'fork'
    from duras2st import DuraS2ST, translate
    from token2wav import Token2wav

    model = DuraS2ST(str(model_path.resolve()))
    try:
        decoder = Token2wav(str(model_path / 'token2wav'))
        text, reasoning, tokens = await translate(model, str(audio), args.target_language)
        wav = decoder(tokens, prompt_wav=str(audio))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(wav)
        if args.show_reasoning:
            print(reasoning)
        print(text)
        print(f'Audio saved to {output}')
    finally:
        await model.shutdown()


if __name__ == '__main__':
    asyncio.run(run(parse_args()))

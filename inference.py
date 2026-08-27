#!/usr/bin/env python3
"""Minimal DuraS2ST inference for English-Chinese speech translation.

The script uses the Step-Audio 2 vLLM backend. DuraS2ST first generates a
duration-planning rationale and then continues generation from that rationale
to produce interleaved translation text and acoustic tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import re
import sys
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

TEXT_TOKEN_VOCAB_SIZE = 151688
AUDIO_TOKEN_OFFSET = 151696
AUDIO_VOCODER_TOKEN_UPPER_BOUND = 6561

THINK_PREFIX = "<think>\n"
THINK_STOP = "</think>"
TTS_FROM_THINK = "\n</think>\n\n<tts_start>"
AUDIO_TOKEN_RE = re.compile(r"<audio_(\d+)>")

SYSTEM_PROMPTS = {
    "zh": "请仔细聆听这段语音，然后将其内容翻译成中文并用语音播报。",
    "en": "请仔细聆听这段中文语音，然后将其内容翻译成英文并用英文语音播报。",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DuraS2ST on one English or Chinese source utterance."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Merged DuraS2ST model directory or Hugging Face repository ID.",
    )
    parser.add_argument(
        "--stepaudio2-root",
        type=Path,
        required=True,
        help="Path to a checkout of https://github.com/stepfun-ai/Step-Audio2.",
    )
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument(
        "--target-language",
        choices=sorted(SYSTEM_PROMPTS),
        required=True,
        help="Target speech language: en or zh.",
    )
    parser.add_argument("--output-audio", type=Path, default=Path("output.wav"))
    parser.add_argument("--show-reasoning", action="store_true")
    parser.add_argument("--max-thinking-tokens", type=int, default=2048)
    parser.add_argument("--max-speech-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser.parse_args()


def resolve_model(model: str) -> Path:
    local = Path(model).expanduser()
    if local.exists():
        return local.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model)).resolve()


def load_stepaudio_modules(stepaudio2_root: Path) -> tuple[Any, Any]:
    root = stepaudio2_root.expanduser().resolve()
    required = (root / "utils.py", root / "token2wav.py")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Invalid --stepaudio2-root; missing: " + ", ".join(missing)
        )
    sys.path.insert(0, str(root))
    from token2wav import Token2wav  # type: ignore
    from utils import load_audio  # type: ignore

    return Token2wav, load_audio


class DuraS2STEngine:
    def __init__(
        self,
        model_path: Path,
        load_audio: Any,
        *,
        gpu_memory_utilization: float,
        max_model_len: int,
        tensor_parallel_size: int,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.sampling_params import RequestOutputKind
        from vllm.v1.engine.async_llm import AsyncLLM

        self.load_audio = load_audio
        self.SamplingParams = SamplingParams
        self.RequestOutputKind = RequestOutputKind
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, padding_side="right"
        )
        self.tokenizer.eos_token = "<|EOT|>"
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|EOT|>")
        engine_args = AsyncEngineArgs(
            model=str(model_path),
            trust_remote_code=True,
            max_model_len=max_model_len,
            max_num_seqs=1,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"audio": 64},
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.engine = AsyncLLM.from_engine_args(engine_args)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text=text, add_special_tokens=False)["input_ids"]

    def build_inputs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_ids: list[int] = []
        audio_inputs: list[tuple[Any, int]] = []

        for message in messages:
            role = "human" if message["role"] == "user" else message["role"]
            content = message["content"]
            prompt_ids.extend(self.encode(f"<|BOT|>{role}\n"))

            if isinstance(content, str):
                prompt_ids.extend(self.encode(content))
            elif isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        prompt_ids.extend(self.encode(item["text"]))
                    elif item["type"] == "audio":
                        audio = self.load_audio(item["audio"], target_rate=16000)
                        for start in range(0, audio.shape[0], 16000 * 25):
                            chunk = audio[start : start + 16000 * 25]
                            if chunk.numel() == 0:
                                continue
                            prompt_ids.extend(self.encode("<audio_patch>"))
                            audio_inputs.append((chunk.numpy(), 16000))
                    else:
                        raise ValueError(f"Unsupported content item: {item['type']}")
            elif content is not None:
                raise TypeError(f"Unsupported message content: {type(content)}")

            if message.get("eot", True):
                prompt_ids.append(self.eos_token_id)

        inputs: dict[str, Any] = {"prompt_token_ids": prompt_ids}
        if audio_inputs:
            inputs["multi_modal_data"] = {"audio": audio_inputs}
        return inputs

    async def _generate_compat(
        self,
        inputs: dict[str, Any],
        sampling: Any,
        request_id: str,
    ) -> AsyncIterator[Any]:
        calls = (
            lambda: self.engine.generate(inputs, sampling, request_id=request_id),
            lambda: self.engine.generate(
                request_id=request_id, prompt=inputs, sampling_params=sampling
            ),
            lambda: self.engine.generate(
                request_id=request_id, inputs=inputs, sampling_params=sampling
            ),
        )
        last_error: TypeError | None = None
        for call in calls:
            try:
                async for output in call():
                    yield output
                return
            except TypeError as error:
                last_error = error
        if last_error is not None:
            raise last_error

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        stop: list[str] | None = None,
    ) -> tuple[str, list[int]]:
        sampling = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop,
            stop_token_ids=[self.eos_token_id],
            skip_special_tokens=False,
            output_kind=self.RequestOutputKind.FINAL_ONLY,
        )
        request_id = f"duras2st-{uuid.uuid4().hex}"
        token_ids: list[int] = []
        raw_text = ""
        async for output in self._generate_compat(
            self.build_inputs(messages), sampling, request_id
        ):
            if output.outputs:
                completion = output.outputs[0]
                token_ids = list(getattr(completion, "token_ids", []) or [])
                raw_text = getattr(completion, "text", "") or ""
            if output.finished:
                break

        if token_ids:
            text_ids = [token for token in token_ids if token < TEXT_TOKEN_VOCAB_SIZE]
            audio_ids = [
                token - AUDIO_TOKEN_OFFSET
                for token in token_ids
                if token >= AUDIO_TOKEN_OFFSET
            ]
            text = self.tokenizer.decode(text_ids)
        else:
            audio_ids = [int(token) for token in AUDIO_TOKEN_RE.findall(raw_text)]
            text = AUDIO_TOKEN_RE.sub("", raw_text)
        return text.replace("<|EOT|>", "").strip(), audio_ids

    async def shutdown(self) -> None:
        result = self.engine.shutdown()
        if inspect.isawaitable(result):
            await result


def clean_reasoning(text: str) -> str:
    return re.sub(r"</think>\s*$", "", text, flags=re.IGNORECASE).strip()


def clean_translation(text: str) -> str:
    text = re.sub(r"</?tts(?:_start|_end)?>", "", text, flags=re.IGNORECASE)
    return text.strip()


async def run(args: argparse.Namespace) -> None:
    if not args.input_audio.is_file():
        raise FileNotFoundError(f"Input audio not found: {args.input_audio}")

    model_path = resolve_model(args.model)
    token2wav_dir = model_path / "token2wav"
    if not token2wav_dir.is_dir():
        raise FileNotFoundError(
            f"Missing token2wav directory in merged checkpoint: {token2wav_dir}"
        )

    Token2wav, load_audio = load_stepaudio_modules(args.stepaudio2_root)
    engine = DuraS2STEngine(
        model_path,
        load_audio,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    token2wav = Token2wav(str(token2wav_dir))

    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[args.target_language]},
        {
            "role": "human",
            "content": [{"type": "audio", "audio": str(args.input_audio.resolve())}],
        },
    ]

    try:
        reasoning, _ = await engine.generate(
            base_messages
            + [{"role": "assistant", "content": THINK_PREFIX, "eot": False}],
            max_tokens=args.max_thinking_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop=[THINK_STOP],
        )
        reasoning = clean_reasoning(reasoning)

        translation, audio_tokens = await engine.generate(
            base_messages
            + [
                {
                    "role": "assistant",
                    "content": f"{THINK_PREFIX}{reasoning}{TTS_FROM_THINK}",
                    "eot": False,
                }
            ],
            max_tokens=args.max_speech_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        translation = clean_translation(translation)
        audio_tokens = [
            token for token in audio_tokens if token < AUDIO_VOCODER_TOKEN_UPPER_BOUND
        ]
        if not audio_tokens:
            raise RuntimeError("The model generated no valid audio tokens.")

        audio_bytes = token2wav(
            audio_tokens, prompt_wav=str(args.input_audio.resolve())
        )
        args.output_audio.parent.mkdir(parents=True, exist_ok=True)
        args.output_audio.write_bytes(audio_bytes)

        if args.show_reasoning:
            print(f"Reasoning: {reasoning}")
        print(f"Translation: {translation}")
        print(f"Audio: {args.output_audio.resolve()}")
    finally:
        await engine.shutdown()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()

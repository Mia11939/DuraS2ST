"""Token generation and termination protocol used for the released model."""
from __future__ import annotations
import asyncio
import copy
import re
import uuid
from dataclasses import dataclass
from functools import wraps
from typing import Any
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
import torchaudio


def load_audio(path, target_rate=16000):
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != target_rate:
        waveform = torchaudio.transforms.Resample(sample_rate, target_rate)(waveform)
    return waveform[0]

TEXT_TOKEN_VOCAB_SIZE = 151688

AUDIO_TOKEN_OFFSET = 151696

AUDIO_VOCODER_TOKEN_UPPER_BOUND = 6561

THINK_PREFIX = "<think>\n"

THINK_STOP = "</think>"

TTS_FROM_THINK = "\n</think>\n\n<tts_start>"

AUDIO_TOKEN_RE = re.compile(r"<audio_(\d+)>")

@dataclass
class GenerationResult:
    output_text: str
    audio_token_ids: list[int]
    finish_reason: str | None

def _configure_audio_lengths():
    from vllm.model_executor.models.mm_step_audio import Step1fProcessor, StepAudio2ForCausalLM
    if getattr(Step1fProcessor, '_duras2st_audio_lengths', False):
        return
    original_count=Step1fProcessor.get_num_audio_tokens
    original_process=StepAudio2ForCausalLM._process_audio_input

    @wraps(original_count)
    def count(self, max_feature_len):
        if max_feature_len < 2:
            raise ValueError('Expected StepAudio2 mel frames including two padding frames')
        return original_count(self,max_feature_len-2)

    @wraps(original_process)
    def process(self, audio_input):
        data=dict(audio_input)
        lengths=data['audio_lens']
        if any(n < 2 for n in lengths):
            raise ValueError('Expected StepAudio2 mel frames including two padding frames')
        # Keep full padded mels. Only mask lengths and resulting audio-token
        # counts change, as required by the model audio encoder.
        data['audio_lens']=[n-2 for n in lengths]
        return original_process(self,data)

    Step1fProcessor.get_num_audio_tokens=count
    StepAudio2ForCausalLM._process_audio_input=process
    Step1fProcessor._duras2st_audio_lengths=True


class DuraS2ST:
    """vLLM engine for DuraS2ST text and speech generation."""

    def __init__(self, model_path: str) -> None:
        _configure_audio_lengths()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="right",
        )
        self.tokenizer.eos_token = "<|EOT|>"
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|EOT|>")
        self.tts_end_token_id = self.tokenizer.convert_tokens_to_ids("<tts_end>")
        self.speech_eos_token_id = self.tokenizer.convert_tokens_to_ids(
            f"<audio_{AUDIO_VOCODER_TOKEN_UPPER_BOUND}>"
        )
        self.audio_token_offset = self.tokenizer.convert_tokens_to_ids("<audio_0>")
        expected = {
            "<|EOT|>": self.eos_token_id,
            "<tts_end>": self.tts_end_token_id,
            "<audio_0>": self.audio_token_offset,
            f"<audio_{AUDIO_VOCODER_TOKEN_UPPER_BOUND}>": self.speech_eos_token_id,
        }
        unk_id = self.tokenizer.unk_token_id
        invalid = {token: token_id for token, token_id in expected.items()
                   if token_id is None or token_id == unk_id}
        if invalid:
            raise ValueError(f"StepAudio2 tokenizer is missing required tokens: {invalid}")
        if self.audio_token_offset != AUDIO_TOKEN_OFFSET:
            raise ValueError(
                f"Unexpected <audio_0> id: {self.audio_token_offset} != {AUDIO_TOKEN_OFFSET}"
            )
        if self.speech_eos_token_id != self.audio_token_offset + AUDIO_VOCODER_TOKEN_UPPER_BOUND:
            raise ValueError(
                "Unexpected speech EOS id: "
                f"{self.speech_eos_token_id} != "
                f"{self.audio_token_offset + AUDIO_VOCODER_TOKEN_UPPER_BOUND}"
            )
        self.tts_stop_token_ids = [
            self.eos_token_id,
            self.tts_end_token_id,
            self.speech_eos_token_id,
        ]

        cfg: dict[str, Any] = {
            "model": model_path,
            "trust_remote_code": True,
            "max_model_len": 8192,
            "max_num_seqs": 32,
            "tensor_parallel_size": 1,
            "limit_mm_per_prompt": {"audio": 64},
            "enforce_eager": True,
            "gpu_memory_utilization": 0.7,
            "dtype": "bfloat16",
        }
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**cfg))

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text=text, add_special_tokens=False)["input_ids"]

    def build_inputs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_ids: list[int] = []
        audio_inputs: list[tuple[Any, int]] = []
        for msg in messages:
            role = "human" if msg["role"] == "user" else msg["role"]
            content = msg["content"]
            eot = msg.get("eot", True)

            if content is None:
                prompt_ids.extend(self._encode(f"<|BOT|>{role}\n"))
                continue
            if isinstance(content, str):
                prompt_ids.extend(self._encode(f"<|BOT|>{role}\n{content}"))
            elif isinstance(content, list):
                prompt_ids.extend(self._encode(f"<|BOT|>{role}\n"))
                for item in content:
                    t = item["type"]
                    if t == "text":
                        prompt_ids.extend(self._encode(item["text"]))
                    elif t == "audio":
                        audio = load_audio(item["audio"], target_rate=16000)
                        # One <audio_patch> per 25-sec chunk; fork expands each itself.
                        for start in range(0, audio.shape[0], 16000 * 25):
                            chunk = audio[start:start + 16000 * 25]
                            if chunk.numel() == 0:
                                continue
                            prompt_ids.extend(self._encode("<audio_patch>"))
                            audio_inputs.append((chunk.numpy(), 16000))
                    elif t == "token":
                        prompt_ids.extend(item["token"])
                    else:
                        raise ValueError(f"Unsupported content type: {t}")
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
            if eot:
                prompt_ids.append(self.eos_token_id)

        llm_inputs: dict[str, Any] = {"prompt_token_ids": prompt_ids}
        if audio_inputs:
            llm_inputs["multi_modal_data"] = {"audio": audio_inputs}
        return llm_inputs

    def _split_generation_tokens(self, output_token_ids: list[int]) -> tuple[list[int], list[int]]:
        """Extract text and codec tokens up to the first speech terminator."""
        text_ids: list[int] = []
        audio_ids: list[int] = []
        terminal_ids = {self.eos_token_id, self.tts_end_token_id, self.speech_eos_token_id}
        for token_id in output_token_ids:
            if token_id in terminal_ids:
                break
            if token_id < TEXT_TOKEN_VOCAB_SIZE:
                text_ids.append(token_id)
                continue
            if token_id < self.audio_token_offset:
                continue
            audio_id = token_id - self.audio_token_offset
            if audio_id >= AUDIO_VOCODER_TOKEN_UPPER_BOUND:
                break
            audio_ids.append(audio_id)
        return text_ids, audio_ids

    async def __call__(self, messages: list[dict[str, Any]], **kwargs: Any,
                       ) -> GenerationResult:
        kwargs.setdefault("skip_special_tokens", False)
        kwargs.setdefault("output_kind", RequestOutputKind.FINAL_ONLY)
        # The official Transformers inference stops on all three IDs. Stopping
        # only on <|EOT|> lets generation run past <tts_end>/speech-EOS and can
        # loop until max_tokens.
        kwargs.setdefault("stop_token_ids", self.tts_stop_token_ids)
        sampling = SamplingParams(**kwargs)
        request_id = f"step-audio2-offline-{uuid.uuid4().hex}"

        output_token_ids: list[int] = []
        raw_text = ""
        finish_reason: str | None = None
        async for out in self.engine.generate(
            self.build_inputs(messages), sampling, request_id=request_id,
        ):
            if out.outputs:
                comp = out.outputs[0]
                output_token_ids = list(getattr(comp, "token_ids", []) or [])
                raw_text = getattr(comp, "text", "") or ""
                finish_reason = getattr(comp, "finish_reason", None)
            if out.finished:
                break

        if output_token_ids:
            text_ids, audio_ids = self._split_generation_tokens(output_token_ids)
            output_text = self.tokenizer.decode(text_ids)
        else:
            # Fallback when token_ids are absent: parse <audio_N> from text.
            audio_ids = [int(m) for m in AUDIO_TOKEN_RE.findall(raw_text)]
            output_text = AUDIO_TOKEN_RE.sub("", raw_text)

        return GenerationResult(
            output_text=output_text.replace("<|EOT|>", "").strip(),
            audio_token_ids=audio_ids,
            finish_reason=finish_reason,
        )

    async def shutdown(self) -> None:
        result = self.engine.shutdown()
        if asyncio.iscoroutine(result):
            await result

async def translate(model, audio_path, target_language):
    """Generate the reasoning, translation, and speech tokens for one utterance."""
    prompt = ('请仔细聆听这段语音，然后将其内容翻译成中文并用语音播报。'
              if target_language == 'zh' else
              '请仔细聆听这段中文语音，然后将其内容翻译成英文并用英文语音播报。')
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'human', 'content': [{'type': 'audio', 'audio': audio_path}]},
        {'role': 'assistant', 'content': THINK_PREFIX, 'eot': False},
    ]
    sampling = dict(max_tokens=4096, temperature=0.7, top_p=0.9,
                    repetition_penalty=1.05, skip_special_tokens=False)
    think = await model(copy.deepcopy(messages), **sampling,
                        stop=[THINK_STOP], stop_token_ids=[model.eos_token_id], seed=42)
    if think.finish_reason == 'length':
        raise RuntimeError('Reasoning did not finish within the token limit')
    reasoning = re.sub(r'</think>\s*$', '', think.output_text, flags=re.IGNORECASE).strip()
    messages[-1]['content'] = f'{THINK_PREFIX}{reasoning}{TTS_FROM_THINK}'
    speech = await model(messages, **sampling, seed=43)
    if speech.finish_reason == 'length':
        raise RuntimeError('Speech generation did not finish within the token limit')
    if not speech.output_text or not speech.audio_token_ids:
        raise RuntimeError('No translation or speech tokens generated')
    return speech.output_text, reasoning, speech.audio_token_ids

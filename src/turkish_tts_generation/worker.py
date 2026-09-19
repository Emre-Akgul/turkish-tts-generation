"""JSON-lines worker used by isolated inference-engine environments."""

import argparse
import contextlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _device(value: str) -> str:
    if value != "auto":
        return value
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _reference(item: dict[str, Any], options: dict[str, Any]) -> str | None:
    value = item.get("reference_audio") or options.get("reference_audio") or options.get("_default_reference_audio")
    return str(value) if value else None


def _release_cuda_cache() -> None:
    with contextlib.suppress(ImportError):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _seed_everything(seed: int) -> None:
    """Seed available backend RNGs without adding worker dependencies."""
    random.seed(seed)
    with contextlib.suppress(ImportError):
        import numpy

        numpy.random.seed(seed)
    with contextlib.suppress(ImportError):
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class Backend:
    sample_rate = 0

    def generate(self, item: dict[str, Any]) -> tuple[int, float | None]:
        raise NotImplementedError


class VoxCPMBackend(Backend):
    def __init__(self, model_path: Path, device: str, options: dict[str, Any], _companion: Path | None) -> None:
        import soundfile as sf
        from voxcpm.core import VoxCPM

        self.sf = sf
        self.options = options
        self.model = VoxCPM.from_pretrained(
            hf_model_id=str(model_path),
            load_denoiser=False,
            optimize=bool(options.get("optimize", True)),
            device=device,
        )
        self.sample_rate = int(self.model.tts_model.sample_rate)

    def generate(self, item: dict[str, Any]) -> tuple[int, float]:
        import numpy as np

        reference = _reference(item, self.options)
        reference_text = item.get("reference_text") or self.options.get("reference_text")
        attempts = int(self.options.get("saturation_retries", 5)) + 1
        base_seed = int(self.options.get("seed", 42))
        saturation_threshold = float(self.options.get("saturation_threshold", 0.5))
        audio = None
        for attempt in range(attempts):
            _seed_everything(base_seed + attempt)
            candidate = self.model.generate(
                text=item["text"],
                prompt_wav_path=reference if reference_text else None,
                prompt_text=reference_text,
                reference_wav_path=reference if not reference_text else None,
                cfg_value=float(self.options.get("cfg_value", 2.0)),
                inference_timesteps=int(self.options.get("inference_timesteps", 10)),
                max_len=int(self.options.get("max_len", 4096)),
                # Preserve benchmark input verbatim unless normalization is explicitly
                # requested. VoxCPM's normalizer treats every non-Chinese language as
                # English and crashes on valid Turkish forms such as "%25".
                normalize=bool(self.options.get("normalize", False)),
                denoise=False,
            )
            clipped_fraction = float(np.mean(np.abs(candidate) >= 0.999))
            if np.isfinite(candidate).all() and clipped_fraction <= saturation_threshold:
                audio = candidate
                break
            print(
                f"Saturated VoxCPM output for {item['sample_id']} "
                f"(fraction={clipped_fraction:.4f}); retrying with seed {base_seed + attempt + 1}",
                file=sys.stderr,
            )
        if audio is None:
            raise RuntimeError(f"VoxCPM remained saturated after {attempts} deterministic attempts")
        self.sf.write(item["output_path"], audio, self.sample_rate)
        return self.sample_rate, len(audio) / self.sample_rate


class ChatterboxBackend(Backend):
    def __init__(self, model_path: Path, device: str, options: dict[str, Any], _companion: Path | None) -> None:
        import tempfile

        import torchaudio
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.torchaudio = torchaudio
        self.options = options
        # chatterbox-tts==0.1.7 hardcodes this filename in from_local(); the
        # published checkpoint ships it as t3_mtl23ls_v3.safetensors instead.
        expected_t3 = model_path / "t3_mtl23ls_v2.safetensors"
        load_path = model_path
        if not expected_t3.exists():
            try:
                expected_t3.symlink_to(model_path / "t3_mtl23ls_v3.safetensors")
            except OSError:
                # FAT/exFAT model drives do not support symlinks. Build a tiny
                # compatibility view on the local filesystem rather than copying
                # the multi-gigabyte checkpoint just to provide its legacy name.
                self._compat_directory = tempfile.TemporaryDirectory(prefix="chatterbox-model-")
                load_path = Path(self._compat_directory.name)
                for child in model_path.iterdir():
                    (load_path / child.name).symlink_to(child, target_is_directory=child.is_dir())
                (load_path / expected_t3.name).symlink_to(model_path / "t3_mtl23ls_v3.safetensors")
        self.model = ChatterboxMultilingualTTS.from_local(str(load_path), device)
        self.sample_rate = int(self.model.sr)

    def generate(self, item: dict[str, Any]) -> tuple[int, float]:
        # The alignment-repetition guard occasionally forces EOS after only a
        # handful of tokens, producing near-silent output; retry with fresh
        # sampling noise rather than shipping a broken clip.
        minimum_seconds = float(self.options.get("minimum_seconds", 0.5))
        attempts = int(self.options.get("silence_retries", 3))
        duration = 0.0
        for _ in range(attempts):
            wav = self.model.generate(
                item["text"],
                language_id=str(self.options.get("language", "tr")),
                audio_prompt_path=_reference(item, self.options),
                exaggeration=float(self.options.get("exaggeration", 0.5)),
                cfg_weight=float(self.options.get("cfg_weight", 0.5)),
                temperature=float(self.options.get("temperature", 0.8)),
                repetition_penalty=float(self.options.get("repetition_penalty", 1.2)),
                min_p=float(self.options.get("min_p", 0.05)),
                top_p=float(self.options.get("top_p", 1.0)),
            )
            duration = wav.shape[-1] / self.sample_rate
            if duration >= minimum_seconds:
                break
        else:
            raise RuntimeError(f"chatterbox produced near-silent audio after {attempts} attempts: {duration:.2f}s")
        self.torchaudio.save(item["output_path"], wav, self.sample_rate)
        return self.sample_rate, wav.shape[-1] / self.sample_rate


class F5Backend(Backend):
    def __init__(self, model_path: Path, device: str, options: dict[str, Any], companion: Path | None) -> None:
        from f5_tts.api import F5TTS

        self.options = options
        self.model = F5TTS(
            ckpt_file=str(model_path / "orkhon_tts.pt"),
            vocab_file=str(model_path / "vocab.txt"),
            vocoder_local_path=str(companion) if companion else None,
            device=device,
        )

    def generate(self, item: dict[str, Any]) -> tuple[int, float]:
        import soundfile as sf

        reference = _reference(item, self.options)
        self.model.infer(
            ref_file=reference,
            ref_text=item.get("reference_text") or self.options.get("reference_text", ""),
            gen_text=item["text"],
            file_wave=item["output_path"],
            nfe_step=int(self.options.get("steps", 64)),
        )
        info = sf.info(item["output_path"])
        return info.samplerate, info.duration


class MossBackend(Backend):
    def __init__(self, model_path: Path, device: str, options: dict[str, Any], companion: Path | None) -> None:
        from moss_tts_nano_runtime import NanoTTSService

        if companion is None:
            raise ValueError("moss-tts requires the audio-tokenizer companion checkpoint")
        self.options = options
        self.service = NanoTTSService(
            checkpoint_path=model_path,
            audio_tokenizer_path=companion,
            device=device,
            dtype=str(options.get("runtime_dtype", "auto")),
            attn_implementation=str(options.get("attention", "auto")),
            output_dir=Path(options.get("temporary_output_dir", ".moss-output")),
        )

    def generate(self, item: dict[str, Any]) -> tuple[int, float | None]:
        reference = _reference(item, self.options)
        result = self.service.synthesize(
            text=item["text"],
            output_audio_path=item["output_path"],
            prompt_audio_path=reference,
            prompt_text=item.get("reference_text") or self.options.get("reference_text"),
            seed=int(self.options.get("seed", 42)),
            mode=str(self.options.get("mode", "voice_clone")),
            max_new_frames=int(self.options.get("max_new_frames", 375)),
        )
        return int(result["sample_rate"]), _float_or_none(result.get("duration_seconds"))


class SupertonicBackend(Backend):
    def __init__(self, model_path: Path, _device: str, options: dict[str, Any], _companion: Path | None) -> None:
        from supertonic import TTS

        self.options = options
        self.model = TTS(model="supertonic-3", model_dir=str(model_path), auto_download=False)
        self.style = self.model.get_voice_style(str(options.get("voice", "F1")))
        self.sample_rate = int(self.model.sample_rate)

    def generate(self, item: dict[str, Any]) -> tuple[int, float | None]:
        wav, duration = self.model.synthesize(
            item["text"],
            voice_style=self.style,
            lang=str(self.options.get("language", "tr")),
            total_steps=int(self.options.get("steps", 8)),
            speed=float(self.options.get("speed", 1.05)),
        )
        self.model.save_audio(wav, item["output_path"])
        return self.sample_rate, _float_or_none(duration)


class XTTSBackend(Backend):
    def __init__(self, model_path: Path, device: str, options: dict[str, Any], _companion: Path | None) -> None:
        from TTS.api import TTS

        self.options = options
        self.model = TTS(
            model_path=str(model_path),
            config_path=str(model_path / "config.json"),
            progress_bar=False,
            gpu=device.startswith("cuda"),
        )

    def generate(self, item: dict[str, Any]) -> tuple[int, float]:
        import soundfile as sf

        kwargs: dict[str, Any] = {
            "text": item["text"],
            "file_path": item["output_path"],
            "language": str(self.options.get("language", "tr")),
        }
        reference = _reference(item, self.options)
        speaker = item.get("speaker_id") or self.options.get("speaker_id")
        if reference:
            kwargs["speaker_wav"] = reference
        elif speaker:
            kwargs["speaker"] = speaker
        else:
            raise FileNotFoundError("XTTS default reference audio is missing; download or configure a reference")
        self.model.tts_to_file(**kwargs)
        info = sf.info(item["output_path"])
        return info.samplerate, info.duration


class OmniVoiceBackend(Backend):
    def __init__(self, model_path: Path, device: str, options: dict[str, Any], _companion: Path | None) -> None:
        import torch
        from omnivoice import OmniVoice

        self.options = options
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = OmniVoice.from_pretrained(str(model_path), device_map=device, dtype=dtype)
        self.sample_rate = int(self.model.sampling_rate)

    def generate(self, item: dict[str, Any]) -> tuple[int, float]:
        import soundfile as sf

        audios = self.model.generate(
            text=item["text"],
            language=str(self.options.get("language", "tr")),
            ref_audio=_reference(item, self.options),
            ref_text=item.get("reference_text") or self.options.get("reference_text"),
            instruct=self.options.get("instruct"),
            num_step=int(self.options.get("steps", 16)),
            speed=float(self.options.get("speed", 1.0)),
        )
        audio = audios[0]
        sf.write(item["output_path"], audio, self.sample_rate)
        return self.sample_rate, len(audio) / self.sample_rate


class FishSpeechBackend(Backend):
    def __init__(self, model_path: Path, device: str, options: dict[str, Any], _companion: Path | None) -> None:
        import torch
        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.dac.inference import load_model as load_decoder_model
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue

        self.options = options
        precision = torch.half if options.get("half") else torch.bfloat16
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=model_path,
            device=device,
            precision=precision,
            compile=bool(options.get("compile", False)),
        )
        decoder_model = load_decoder_model(
            config_name=str(options.get("decoder_config_name", "modded_dac_vq")),
            checkpoint_path=model_path / "codec.pth",
            device=device,
        )
        self.engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            compile=bool(options.get("compile", False)),
            precision=precision,
        )

    def generate(self, item: dict[str, Any]) -> tuple[int, float]:
        import soundfile as sf
        from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

        reference = _reference(item, self.options)
        reference_text = item.get("reference_text") or self.options.get("reference_text")
        references = []
        if reference and reference_text:
            references = [ServeReferenceAudio(audio=Path(reference).read_bytes(), text=reference_text)]
        request = ServeTTSRequest(
            text=item["text"],
            references=references,
            seed=int(self.options.get("seed", 42)),
            max_new_tokens=int(self.options.get("max_new_tokens", 1024)),
            chunk_length=int(self.options.get("chunk_length", 200)),
            top_p=float(self.options.get("top_p", 0.8)),
            repetition_penalty=float(self.options.get("repetition_penalty", 1.1)),
            temperature=float(self.options.get("temperature", 0.8)),
        )
        results = list(self.engine.inference(request))
        final = next((result for result in results if result.code == "final"), None)
        if final is None or final.audio is None:
            error = next((result.error for result in results if result.code == "error"), None)
            raise RuntimeError(str(error) if error else "fish-speech produced no audio")
        sample_rate, audio = final.audio
        sf.write(item["output_path"], audio, sample_rate)
        return sample_rate, len(audio) / sample_rate


class FreyaBackend(Backend):
    def __init__(self, model_path: Path, device: str, options: dict[str, Any], companion: Path | None) -> None:
        if companion is None:
            raise ValueError("freya requires the VoxCPM2 AudioVAE companion checkpoint")
        import freyatts.vae
        from freyatts import FreyaTTS

        audiovae_path = companion / "audiovae.pth"
        original_download = freyatts.vae.hf_hub_download
        freyatts.vae.hf_hub_download = lambda *_args, **_kwargs: str(audiovae_path)
        try:
            self.model = FreyaTTS.from_pretrained(str(model_path), device=device)
        finally:
            freyatts.vae.hf_hub_download = original_download
        self.options = options
        self.sample_rate = int(self.model.sample_rate)

    def generate(self, item: dict[str, Any]) -> tuple[int, float]:
        wav = self.model.synthesize(
            item["text"],
            steps=int(self.options.get("steps", 32)),
            seed=int(self.options.get("seed", 9)),
        )
        self.model.save_wav(wav, item["output_path"])
        return self.sample_rate, len(wav) / self.sample_rate


class PiperBackend(Backend):
    def __init__(self, model_path: Path, device: str, options: dict[str, Any], _companion: Path | None) -> None:
        from piper import PiperVoice

        self.options = options
        onnx_path = next(model_path.rglob("*.onnx"))
        self.voice = PiperVoice.load(str(onnx_path), use_cuda=device.startswith("cuda"))
        self.sample_rate = self.voice.config.sample_rate

    def generate(self, item: dict[str, Any]) -> tuple[int, float]:
        import wave

        from piper.config import SynthesisConfig

        syn_config = SynthesisConfig(
            length_scale=float(self.options.get("length_scale", 1.0)),
            noise_scale=float(self.options.get("noise_scale", 0.667)),
            noise_w_scale=float(self.options.get("noise_w_scale", 0.8)),
        )
        with wave.open(item["output_path"], "wb") as wav_file:
            self.voice.synthesize_wav(item["text"], wav_file, syn_config=syn_config)
        import soundfile as sf

        info = sf.info(item["output_path"])
        return info.samplerate, info.duration


class MMSBackend(Backend):
    def __init__(self, model_path: Path, device: str, options: dict[str, Any], _companion: Path | None) -> None:
        import torch
        from transformers import AutoTokenizer, VitsModel

        self.options = options
        self.torch = torch
        self.device = device
        self.model = VitsModel.from_pretrained(str(model_path)).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.sample_rate = int(self.model.config.sampling_rate)

    def generate(self, item: dict[str, Any]) -> tuple[int, float]:
        import soundfile as sf

        inputs = self.tokenizer(item["text"], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            waveform = self.model(**inputs).waveform
        audio = waveform.squeeze().cpu().float().numpy()
        sf.write(item["output_path"], audio, self.sample_rate)
        return self.sample_rate, len(audio) / self.sample_rate


class AnkaBackend(Backend):
    sample_rate = 24000

    def __init__(self, model_path: Path, device: str, options: dict[str, Any], _companion: Path | None) -> None:
        from anka import AnkaTTS

        self.options = options
        self.model = AnkaTTS.from_pretrained(str(model_path), device=device)

    def generate(self, item: dict[str, Any]) -> tuple[int, float]:
        reference = _reference(item, self.options)
        reference_text = item.get("reference_text") or self.options.get("reference_text")
        kwargs: dict[str, Any] = {"seed": int(self.options.get("seed", 42))}
        if self.options.get("speed") is not None:
            kwargs["speed"] = float(self.options["speed"])
        if self.options.get("nfe_step") is not None:
            kwargs["nfe_step"] = int(self.options["nfe_step"])
        if self.options.get("cfg_strength") is not None:
            kwargs["cfg_strength"] = float(self.options["cfg_strength"])
        if reference:
            kwargs["ref_audio"] = reference
            kwargs["ref_text"] = reference_text
        else:
            kwargs["voice"] = str(self.options.get("voice", "male"))
        wav = self.model.synthesize(item["text"], **kwargs)
        self.model.save_wav(wav, item["output_path"])
        return self.sample_rate, len(wav) / self.sample_rate


class PocketBackend(Backend):
    def __init__(self, model_path: Path, device: str, options: dict[str, Any], _companion: Path | None) -> None:
        import yaml
        from pocket_tts import TTSModel

        self.options = options
        raw_config = yaml.safe_load((model_path / "config.yaml").read_text(encoding="utf-8"))
        raw_config["weights_path"] = str(model_path / "model.safetensors")
        raw_config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(model_path / "tokenizer.model")
        patched_config = model_path / "_patched_config.yaml"
        patched_config.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
        self.model = TTSModel.load_model(
            config=str(patched_config),
            temp=options.get("temperature"),
            sampler_decode_steps=int(options.get("sampler_decode_steps", 1)),
            quantize=bool(options.get("quantize", False)),
        )
        self.model.to(device)
        self.sample_rate = int(self.model.sample_rate)
        self._pcm16_cache: dict[str, str] = {}

    def _as_pcm16(self, path: str) -> str:
        # pocket-tts's own WAV reader is the stdlib `wave` module, which raises on
        # non-integer PCM (e.g. our float32 reference clips) before it ever reaches
        # its soundfile fallback. Re-encode once per process and reuse the copy.
        cached = self._pcm16_cache.get(path)
        if cached:
            return cached
        import tempfile

        import soundfile as sf

        if sf.info(path).subtype == "PCM_16":
            self._pcm16_cache[path] = path
            return path
        data, sample_rate = sf.read(path)
        descriptor, converted = tempfile.mkstemp(suffix=".wav")
        os.close(descriptor)
        sf.write(converted, data, sample_rate, subtype="PCM_16")
        self._pcm16_cache[path] = converted
        return converted

    def generate(self, item: dict[str, Any]) -> tuple[int, float]:
        import soundfile as sf

        reference = _reference(item, self.options)
        if not reference:
            raise FileNotFoundError("pocket-tts requires a reference audio prompt")
        state = self.model.get_state_for_audio_prompt(self._as_pcm16(reference))
        frames_after_eos = self.options.get("frames_after_eos")
        audio = self.model.generate_audio(
            state,
            item["text"],
            frames_after_eos=int(frames_after_eos) if frames_after_eos is not None else None,
        )
        waveform = audio.squeeze(0).detach().cpu().numpy()
        sf.write(item["output_path"], waveform, self.sample_rate)
        return self.sample_rate, float(waveform.shape[-1]) / self.sample_rate


class KaniBackend(Backend):
    def __init__(self, model_path: Path, device: str, options: dict[str, Any], _companion: Path | None) -> None:
        from kani_tts import KaniTTS

        self.options = options
        self.model = KaniTTS(str(model_path), device_map=device, show_info=False)
        self.sample_rate = int(self.model.sample_rate)

    def generate(self, item: dict[str, Any]) -> tuple[int, float]:
        import soundfile as sf

        speaker = item.get("speaker_id") or self.options.get("speaker_id")
        audio, _ = self.model.generate(
            item["text"],
            speaker_id=speaker,
            temperature=float(self.options.get("temperature", 1.0)),
            top_p=float(self.options.get("top_p", 0.95)),
            repetition_penalty=float(self.options.get("repetition_penalty", 1.1)),
        )
        sf.write(item["output_path"], audio, self.sample_rate)
        return self.sample_rate, float(len(audio)) / self.sample_rate


class HiggsBackend(Backend):
    """Higgs TTS 3 ships weights only; the model card's own AGENTS.md directs
    self-hosting through an SGLang-Omni server exposing an OpenAI-compatible
    /v1/audio/speech endpoint. This backend manages that server as a subprocess.
    """

    sample_rate = 24000

    def __init__(self, model_path: Path, device: str, options: dict[str, Any], _companion: Path | None) -> None:
        import atexit
        import subprocess

        import requests

        self.options = options
        self.requests = requests
        self.port = int(options.get("port", 8000))
        self.base_url = f"http://127.0.0.1:{self.port}"
        command = ["sgl-omni", "serve", "--model-path", str(model_path), "--port", str(self.port)]
        self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)  # noqa: S603
        atexit.register(self._terminate)
        deadline = time.monotonic() + float(options.get("startup_timeout_seconds", 900))
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("sgl-omni server exited during startup")
            try:
                if requests.get(f"{self.base_url}/health", timeout=2).ok:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        else:
            self._terminate()
            raise RuntimeError("sgl-omni server did not become ready in time")

    def generate(self, item: dict[str, Any]) -> tuple[int, float]:
        import soundfile as sf

        reference = _reference(item, self.options)
        reference_text = item.get("reference_text") or self.options.get("reference_text")
        payload: dict[str, Any] = {
            "input": item["text"],
            "temperature": float(self.options.get("temperature", 0.8)),
            "top_k": int(self.options.get("top_k", 50)),
            "max_new_tokens": int(self.options.get("max_new_tokens", 1024)),
        }
        if reference:
            payload["references"] = [{"audio_path": reference, "text": reference_text or ""}]
        response = self.requests.post(f"{self.base_url}/v1/audio/speech", json=payload, timeout=300)
        response.raise_for_status()
        Path(item["output_path"]).write_bytes(response.content)
        info = sf.info(item["output_path"])
        return info.samplerate, info.duration

    def _terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self.process.kill()


class FireRedBackend(Backend):
    def __init__(self, model_path: Path, device: str, options: dict[str, Any], _companion: Path | None) -> None:
        from fireredtts3.core import FireRedTTS3

        self.options = options
        self.tts = FireRedTTS3(
            str(model_path),
            use_wetext=bool(options.get("use_wetext", True)),
            use_llm_tn=bool(options.get("use_llm_tn", False)),
        )

    def generate(self, item: dict[str, Any]) -> tuple[int, float]:
        import torchaudio

        reference = _reference(item, self.options)
        if not reference:
            raise FileNotFoundError("firered requires a reference audio prompt")
        reference_text = item.get("reference_text") or self.options.get("reference_text") or ""
        prompt_audio, prompt_audio_sr = torchaudio.load(reference)
        gen_audio, gen_audio_sr = self.tts.generate(
            language=self.options.get("language"),
            prompt_text=reference_text,
            prompt_audio=prompt_audio,
            prompt_audio_sr=prompt_audio_sr,
            text=item["text"],
            do_tn=bool(self.options.get("do_tn", True)),
        )
        torchaudio.save(item["output_path"], gen_audio.cpu(), gen_audio_sr)
        return gen_audio_sr, float(gen_audio.shape[-1]) / gen_audio_sr


class MossV15Backend(Backend):
    def __init__(self, model_path: Path, device: str, options: dict[str, Any], _companion: Path | None) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.options = options
        self.torch = torch
        self.device = device
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(device)
        self.model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True, torch_dtype=dtype).to(device)
        self.model.eval()

    def generate(self, item: dict[str, Any]) -> tuple[int, float]:
        import torchaudio

        reference = _reference(item, self.options)
        kwargs: dict[str, Any] = {"text": item["text"], "language": self.options.get("language", "Turkish")}
        if reference:
            kwargs["reference"] = [reference]
        message = self.processor.build_user_message(**kwargs)
        batch = self.processor([[message]], mode="generation")
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(self.options.get("max_new_tokens", 4096)),
                do_sample=True,
                audio_temperature=float(self.options.get("audio_temperature", 1.7)),
                audio_top_p=float(self.options.get("audio_top_p", 0.8)),
                audio_top_k=int(self.options.get("audio_top_k", 25)),
                audio_repetition_penalty=float(self.options.get("audio_repetition_penalty", 1.0)),
            )
        message_out = next(message for message in self.processor.decode(outputs) if message is not None)
        audio = message_out.audio_codes_list[0]
        sample_rate = int(self.processor.model_config.sampling_rate)
        torchaudio.save(item["output_path"], audio, sample_rate)
        return sample_rate, float(audio.shape[-1]) / sample_rate


BACKENDS = {
    "voxcpm": VoxCPMBackend,
    "chatterbox": ChatterboxBackend,
    "f5-tts": F5Backend,
    "moss-tts": MossBackend,
    "supertonic": SupertonicBackend,
    "xtts": XTTSBackend,
    "omnivoice": OmniVoiceBackend,
    "freya": FreyaBackend,
    "fish-speech": FishSpeechBackend,
    "piper": PiperBackend,
    "mms-tts": MMSBackend,
    "anka-tts": AnkaBackend,
    "pocket-tts": PocketBackend,
    "kani-tts": KaniBackend,
    "higgs": HiggsBackend,
    "firered": FireRedBackend,
    "moss-tts-v1.5": MossV15Backend,
}


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(value[0])  # type: ignore[index]


def _serve(backend: Backend) -> None:
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("command") == "shutdown":
            return
        results = []
        for item in request["items"]:
            Path(item["output_path"]).parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    sample_rate, duration = backend.generate(item)
                results.append(
                    {
                        "sample_id": item["sample_id"],
                        "sample_rate": sample_rate,
                        "duration_seconds": duration,
                        "inference_seconds": time.monotonic() - started,
                        "error": None,
                    }
                )
            except Exception as error:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                results.append({"sample_id": item["sample_id"], "error": f"{type(error).__name__}: {error}"})
            finally:
                _release_cuda_cache()
        print(json.dumps({"results": results}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--companion-path", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--options", default="{}")
    args = parser.parse_args()
    options = json.loads(args.options)
    _seed_everything(int(options.get("seed", 42)))
    try:
        with contextlib.redirect_stdout(sys.stderr):
            backend = BACKENDS[args.engine](args.model_path, _device(args.device), options, args.companion_path)
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"ready": False, "error": f"{type(error).__name__}: {error}"}), flush=True)
        return
    _serve(backend)


if __name__ == "__main__":
    main()

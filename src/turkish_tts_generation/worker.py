"""JSON-lines worker used by isolated inference-engine environments."""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any


def _device(value: str) -> str:
    if value != "auto":
        return value
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _reference(item: dict[str, Any], options: dict[str, Any]) -> str | None:
    value = item.get("reference_audio") or options.get("reference_audio")
    return str(value) if value else None


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
        reference = _reference(item, self.options)
        reference_text = item.get("reference_text") or self.options.get("reference_text")
        audio = self.model.generate(
            text=item["text"],
            prompt_wav_path=reference if reference_text else None,
            prompt_text=reference_text,
            reference_wav_path=reference if not reference_text else None,
            cfg_value=float(self.options.get("cfg_value", 2.0)),
            inference_timesteps=int(self.options.get("inference_timesteps", 16)),
            max_len=int(self.options.get("max_len", 4096)),
            normalize=bool(self.options.get("normalize", True)),
            denoise=False,
            seed=int(self.options.get("seed", 42)),
        )
        self.sf.write(item["output_path"], audio, self.sample_rate)
        return self.sample_rate, len(audio) / self.sample_rate


class ChatterboxBackend(Backend):
    def __init__(self, model_path: Path, device: str, options: dict[str, Any], _companion: Path | None) -> None:
        import torchaudio
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.torchaudio = torchaudio
        self.options = options
        self.model = ChatterboxMultilingualTTS.from_local(
            str(model_path), device=device, t3_model="t3_mtl23ls_v3.safetensors"
        )
        self.sample_rate = int(self.model.sr)

    def generate(self, item: dict[str, Any]) -> tuple[int, float]:
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
        if not reference:
            raise ValueError("f5-tts requires reference_audio on the dataset row or target options")
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
        if not reference:
            raise ValueError("moss-tts requires reference_audio on the dataset row or target options")
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
            raise ValueError("xtts requires reference_audio or speaker_id")
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


BACKENDS = {
    "voxcpm": VoxCPMBackend,
    "chatterbox": ChatterboxBackend,
    "f5-tts": F5Backend,
    "moss-tts": MossBackend,
    "supertonic": SupertonicBackend,
    "xtts": XTTSBackend,
    "omnivoice": OmniVoiceBackend,
    "freya": FreyaBackend,
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
                results.append({"sample_id": item["sample_id"], "error": f"{type(error).__name__}: {error}"})
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
    try:
        with contextlib.redirect_stdout(sys.stderr):
            backend = BACKENDS[args.engine](
                args.model_path, _device(args.device), json.loads(args.options), args.companion_path
            )
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"ready": False, "error": f"{type(error).__name__}: {error}"}), flush=True)
        return
    _serve(backend)


if __name__ == "__main__":
    main()

"""
Arabic text-to-speech with Habibi-TTS (https://github.com/SWivid/Habibi-TTS), on CPU.

    uv run tts/habibi_synth.py "السلام عليكم، كيداير؟" -o reply.wav
    uv run tts/habibi_synth.py --dialect EGY "إزيك عامل إيه؟" -o reply.wav
    uv run tts/habibi_synth.py --ref-audio me.wav --ref-text "transcript of me.wav" "..."

Habibi clones the voice of a short reference clip (clipped to 12 s) and needs its exact
transcript. A reference in the same dialect as the text gives the best results. Without
--ref-audio, the sample bundled with habibi-tts for that dialect is used.

(Not named habibi_tts.py: that would shadow the habibi_tts package.)
"""
import argparse
import os
import sys
import tempfile
from importlib.resources import files

import numpy as np
import soundfile as sf
import truststore

# Verify HTTPS with the Windows certificate store, so model downloads work behind a
# TLS-inspecting corporate proxy (otherwise: SSL CERTIFICATE_VERIFY_FAILED).
truststore.inject_into_ssl()

# f5_tts prints the Arabic reference text, which crashes on a Windows cp1252 console or pipe.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

SR = 24000     # Habibi (vocos vocoder) output rate
PCM_SR = 16000  # rate the rest of the pipeline uses (see ttfa_harness.SR)

DIALECTS = ["UNK", "MSA", "SAU", "UAE", "ALG", "IRQ", "EGY", "MAR", "OMN", "TUN", "LEV", "SDN", "LBY"]

# Reference clips shipped in habibi_tts/assets, with transcripts from its assets/README.md.
# Dialects without a sample (OMN, TUN, LEV, SDN, LBY, UNK) fall back to the MSA one.
BUNDLED_REFS = {
    "MSA": ("MSA.mp3", "كان اللعيب حاضرًا في العديد من الأنشطة والفعاليات المرتبطة بكأس العالم، مما سمح للجماهير بالتفاعل معه والتقاط الصور التذكارية."),
    "SAU": ("Najdi.wav", "تكفى طمني انا اليوم ماني بنايم ولا هو بداخل عيني النوم الين اتطمن عليه."),
    "UAE": ("UAE.wav", "قمنا نشتريها بشكل متكرر أو لما نلقى ستايل يعجبنا وحياناً هذا الستايل ما نحبه."),
    "ALG": ("ALG.wav", "أنيا هكا باغية ناكل هكا أني ن نشوف فيها الحاجة هذيكا."),
    "IRQ": ("IRQ.wav", "يعني ااا ما نقدر ناخذ وقت أكثر، ااا لأنه شروط كلش يحتاجلها وقت."),
    "EGY": ("EGY.mp3", "ايه الكلام. بقولك ايه. استخدم صوتي في المحادثات. استخدمه هيعجبك اوي."),
    "MAR": ("MAR.mp3", "إذا بغيتي شي صوت باللهجة المغربية للإعلانات ديالك هذا أحسن واحد غادي تلقاه."),
}


class HabibiTTS:
    """Loads the Unified Habibi model once; synthesize() can then be called repeatedly."""

    def __init__(self, dialect="MAR", ref_audio=None, ref_text=None, device="cpu", nfe_step=32):
        # Heavy imports here so importing this module stays cheap.
        from cached_path import cached_path
        from f5_tts.infer.utils_infer import load_model, load_vocoder, preprocess_ref_audio_text
        from habibi_tts.model.utils import dialect_id_map
        from hydra.utils import get_class
        from omegaconf import OmegaConf

        if dialect not in DIALECTS:
            raise ValueError(f"unknown dialect {dialect!r}, expected one of {DIALECTS}")
        if ref_audio is None:
            name, ref_text = BUNDLED_REFS.get(dialect, BUNDLED_REFS["MSA"])
            ref_audio = str(files("habibi_tts").joinpath(f"assets/{name}"))
        elif not ref_text:
            raise ValueError("ref_text is required with ref_audio (it must match what the clip says)")

        self.device = device
        self.nfe_step = nfe_step
        self.dialect_id = dialect_id_map[dialect]

        cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
        self.mel_spec_type = cfg.model.mel_spec.mel_spec_type
        self.model = load_model(
            get_class(f"f5_tts.model.{cfg.model.backbone}"),
            cfg.model.arch,
            str(cached_path("hf://SWivid/Habibi-TTS/Unified/model_200000.safetensors")),
            mel_spec_type=self.mel_spec_type,
            vocab_file=str(cached_path("hf://SWivid/Habibi-TTS/Unified/vocab.txt")),
            device=device,
        )
        self.vocoder = load_vocoder(vocoder_name=self.mel_spec_type, device=device)

        # pydub (used by preprocess_ref_audio_text) needs ffmpeg for anything but WAV,
        # so decode with soundfile and hand it a WAV instead.
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            audio, sr = sf.read(ref_audio, dtype="float32")
            sf.write(wav_path, audio, sr)
            self.ref_audio, self.ref_text = preprocess_ref_audio_text(wav_path, ref_text)
        finally:
            os.remove(wav_path)

    def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Returns mono float32 audio at SR (24 kHz)."""
        from habibi_tts.infer.utils_infer import infer_process

        wave, _, _ = infer_process(
            self.ref_audio,
            self.ref_text,
            text,
            self.model,
            self.vocoder,
            mel_spec_type=self.mel_spec_type,
            nfe_step=self.nfe_step,
            speed=speed,
            device=self.device,
            dialect_id=self.dialect_id,
        )
        return wave

    def synthesize_pcm(self, text: str, speed: float = 1.0) -> bytes:
        """Returns 16 kHz mono int16 PCM bytes, the format the pipeline adapters stream."""
        import torch
        import torchaudio.functional as F

        wave = torch.from_numpy(self.synthesize(text, speed)).float()
        wave = F.resample(wave, SR, PCM_SR).numpy()
        return (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", help="text to speak")
    parser.add_argument("-o", "--out", default="habibi_out.wav", help="output WAV path (24 kHz)")
    parser.add_argument("-d", "--dialect", default="MAR", choices=DIALECTS)
    parser.add_argument("--ref-audio", help="reference clip whose voice to clone (default: bundled sample)")
    parser.add_argument("--ref-text", help="exact transcript of --ref-audio")
    parser.add_argument("--nfe-step", type=int, default=32, help="denoising steps; 16 is faster on CPU")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    tts = HabibiTTS(args.dialect, args.ref_audio, args.ref_text, nfe_step=args.nfe_step)
    sf.write(args.out, tts.synthesize(args.text, args.speed), SR)
    print(args.out)


if __name__ == "__main__":
    main()

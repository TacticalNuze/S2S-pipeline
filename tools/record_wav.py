"""
Record 16 kHz mono int16 WAV files from the microphone, for asr/moulsot_asr.py, full-run/habibi_adapter.py and benchmark/ttfa_harness.py.

Usage:
    uv run tools/record_wav.py                     # press Enter to stop, saves recordings/rec_<timestamp>.wav
    uv run tools/record_wav.py -o hello.wav -s 5   # record exactly 5 seconds
    uv run tools/record_wav.py --list-devices      # show microphones, then pick one with -d <index>
"""
import argparse
import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 16000


def record_until_enter(device) -> np.ndarray:
    chunks: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        chunks.put(indata.copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="int16", device=device, callback=callback):
        input("Recording... press Enter to stop.")

    blocks = []
    while not chunks.empty():
        blocks.append(chunks.get())
    return np.concatenate(blocks)[:, 0] if blocks else np.zeros(0, dtype=np.int16)


def record_seconds(seconds: float, device) -> np.ndarray:
    print(f"Recording for {seconds:g} s...")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="int16", device=device)
    sd.wait()
    return audio[:, 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", help="output WAV path (default: recordings/rec_<timestamp>.wav)")
    parser.add_argument("-s", "--seconds", type=float, help="fixed duration; omit to stop with Enter")
    parser.add_argument("-d", "--device", type=int, help="input device index (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    output = Path(args.output or f"recordings/rec_{time.strftime('%Y%m%d_%H%M%S')}.wav")
    output.parent.mkdir(parents=True, exist_ok=True)

    audio = record_seconds(args.seconds, args.device) if args.seconds else record_until_enter(args.device)
    if audio.size == 0:
        print("Nothing recorded.")
        return

    sf.write(output, audio, SR, subtype="PCM_16")
    peak = int(np.abs(audio.astype(np.int32)).max())
    print(f"Saved {output} ({audio.size / SR:.1f} s, peak {peak}/32767)")
    if peak < 1000:
        print("Warning: very quiet recording. Check the microphone or pick another one with -d.")


if __name__ == "__main__":
    main()

import sys

import truststore
from qwen_asr import Qwen3ASRModel

# Verify HTTPS with the Windows certificate store (corporate proxy breaks model downloads otherwise).
truststore.inject_into_ssl()

model = Qwen3ASRModel.from_pretrained(
    "atlasia/moulsot.v0.3",
    dtype="float32",
    device_map="cpu",
)

for path in sys.argv[1:] or ["your_audio.wav"]:
    result = model.transcribe(audio=path, language="Arabic")   # one result per input audio
    print(f"{path}: {result[0].text}")

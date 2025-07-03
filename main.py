from speechbrain.inference.ASR import EncoderDecoderASR
import torch
import time
import logging
import whisper
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"🚀 실행 디바이스: {device}")

def transcribe_wav(wav_path):
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-conformer-transformerlm-ksponspeech",
        savedir="pretrained_models/asr-conformer-transformerlm-ksponspeech",
        run_opts={"device": "cuda"}
    )

    logging.info(f"🎤 음성 파일 분석 시작: {wav_path}")
    start = time.time()

    txt = asr_model.transcribe_file(wav_path)

    elapsed = time.time() - start
    logging.info(f"✅ 분석 완료 (걸린 시간: {elapsed:.2f}초)")
    return txt



def transcribe_with_whisper(audio_path: str) -> str:
    # Whisper 모델 로드 (base, small, medium, large 가능)
    model = whisper.load_model("medium", device=device)  # 또는 "small", "medium", "large"

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_path}")

    logging.info(f"🎤 Whisper 분석 시작: {audio_path}")
    start = time.time()

    result = model.transcribe(audio_path, language="ko")

    elapsed = time.time() - start
    logging.info(f"✅ Whisper 분석 완료 (걸린 시간: {elapsed:.2f}초)")
    return result["text"]  # 자막 텍스트 반환

if __name__ == "__main__":
    wav = "000020.wav"
    # subtitle = transcribe_wav(wav)
    subtitle = transcribe_with_whisper(wav)
    print("📜 Subtitle:\n", subtitle)

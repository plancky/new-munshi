import os

import modal
from modal import enter, method
from modal import exit as modal_exit

from ..core.app import app
from ..core.config import MODEL_DIR, RAW_AUDIO_DIR, TRANSCRIPTIONS_DIR, get_logger
from ..core.images import nemo_image
from ..core.volumes import audio_storage_vol, transcriptions_vol

logger = get_logger(__name__)


with nemo_image.imports():
    import time

    import ffmpeg
    import nemo.collections.asr as nemo_asr
    import soundfile as sf
    import torch


@app.cls(
    image=nemo_image,
    gpu="L40S",
    volumes={
        str(RAW_AUDIO_DIR): audio_storage_vol,
        str(TRANSCRIPTIONS_DIR): transcriptions_vol,
    },
    memory=200000,
    scaledown_window=2,
    timeout=5000,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    max_containers=5,
)
@modal.concurrent(max_inputs=1, target_inputs=1)
class Parakeet:
    # Standard params
    SAMPLE_RATE = 32000
    CHUNK_SECONDS = 60
    OVERLAP_SECONDS = 2
    WINDOW_SIZE = 128
    TRANSCRIBE_BATCH_SIZE = 128
    TRANSCRIBE_WORKERS = 2

    @enter(snap=True)
    def setup(self):

        self.COMPUTE_TYPE = torch.bfloat16

        # Prefer local restored model saved during image build
        import logging
        import os

        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        local_model_path = os.path.join(MODEL_DIR, "parakeet-tdt-0.6b-v3.nemo")
        if os.path.exists(local_model_path):
            # Restore ASR
            self.asr_model = nemo_asr.models.ASRModel.restore_from(
                local_model_path, strict=False
            )
        else:
            self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
                "nvidia/parakeet-tdt-0.6b-v3"
            )

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.asr_model.to(device)
            # Force BF16 weights/activations where supported
            try:
                self.asr_model.to(dtype=self.COMPUTE_TYPE)
            except Exception:
                logger.info("🔍 Failed to set BF16 dtype in ASR model")
                pass
            self.asr_model.eval()
        except Exception:
            pass

        # Force greedy decoding for speed and lower VRAM
        try:
            if getattr(self.asr_model.cfg, "decoding", None) is not None:
                if (
                    getattr(self.asr_model.cfg.decoding, "strategy", None)
                    != "greedy_batch"
                ):
                    logger.info("🔍 Setting greedy decoding strategy")
                    self.asr_model.cfg.decoding.strategy = "greedy_batch"
            self.asr_model.change_decoding_strategy(self.asr_model.cfg.decoding)
        except Exception:
            pass
        self.asr_model.change_attention_model(
            self_attention_model="rel_pos_local_attn", att_context_size=[128, 128]
        )

        # Diarization model removed; Parakeet runs ASR-only

    # Helpers
    def _read_resampled_chunk(self, f, start_frame: int, frames_to_read: int):
        f.seek(start_frame)
        audio_chunk = f.read(frames_to_read, dtype="float32", always_2d=False)
        if audio_chunk is None or len(audio_chunk) == 0:
            return None
        if hasattr(audio_chunk, "ndim") and audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)
        # We pre-convert the input to target_sr via ffmpeg; no resample here
        return audio_chunk

    def _write_pcm16(self, path: str, audio, sr: int):
        sf.write(path, audio, sr, subtype="PCM_16")

    # Helper: convert source to mono WAV and log timing
    def _convert_to_wav(
        self, audio_file_path: str, sample_rate: int, tmp_parent: str, base_name: str
    ) -> str:
        t_open0 = time.time()
        tmp_input_wav = os.path.join(tmp_parent, f"input_{base_name}.wav")
        try:
            (
                ffmpeg.input(audio_file_path)
                .output(
                    tmp_input_wav,
                    ac=1,
                    ar=sample_rate,
                    **{"map_metadata": "-1"},
                    vn=None,
                    acodec="pcm_s16le",
                    f="wav",
                )
                .overwrite_output()
                .global_args("-nostdin", "-loglevel", "error")
                .run(capture_stdout=True, capture_stderr=True)
            )
        except Exception as e:
            err_msg = None
            try:
                err_msg = getattr(e, "stderr", None)
                if err_msg and not isinstance(err_msg, str):
                    try:
                        err_msg = err_msg.decode("utf-8", "ignore")
                    except Exception:
                        err_msg = str(err_msg)
            except Exception:
                pass
            logger.error(
                f"ffmpeg conversion failed for {audio_file_path}: {err_msg or str(e)}"
            )
            raise RuntimeError("ffmpeg conversion failed; see logs for details")
        t_open1 = time.time()
        logger.info(f"🎧 Converted to WAV in {t_open1 - t_open0:.2f}s: {tmp_input_wav}")
        return tmp_input_wav

    # Helper: read PCM and produce chunk files; returns paths and total I/O time
    def _produce_chunks(
        self,
        tmp_input_wav: str,
        sample_rate: int,
        chunk_duration_sec: int,
        overlap_sec: int,
        chunk_dir: str,
    ):
        f = sf.SoundFile(tmp_input_wav, mode="r")
        sr_in = f.samplerate
        total_frames = len(f)
        chunk_frames_in = int(chunk_duration_sec * sr_in)
        overlap_frames_in = int(overlap_sec * sr_in)
        step_in = max(1, chunk_frames_in - overlap_frames_in)

        chunk_paths = []
        idx = 0
        io_time_total = 0.0
        try:
            while True:
                start_frame = idx * step_in
                if start_frame >= total_frames:
                    break
                frames_to_read = min(chunk_frames_in, total_frames - start_frame)
                io_step_t0 = time.time()
                audio_chunk = self._read_resampled_chunk(f, start_frame, frames_to_read)
                if audio_chunk is None or len(audio_chunk) == 0:
                    break
                chunk_path = os.path.join(chunk_dir, f"chunk_{idx:05d}.wav")
                self._write_pcm16(chunk_path, audio_chunk, sample_rate)
                chunk_paths.append(chunk_path)
                idx += 1
                io_time_total += time.time() - io_step_t0
        finally:
            try:
                f.close()
            except Exception:
                pass
        return chunk_paths, io_time_total

    # Helper: transcribe over windows and tail (diarization removed)
    def _process_windows(self, chunk_paths, enable_speakers: bool):
        collected_outputs = []
        collected_segments = []
        transcribe_time_total = 0.0
        diar_time_total = 0.0

        if len(chunk_paths) >= self.WINDOW_SIZE:
            for start in range(
                self.WINDOW_SIZE, len(chunk_paths) + 1, self.WINDOW_SIZE
            ):
                window = chunk_paths[start - self.WINDOW_SIZE : start]
                t0 = time.time()
                outputs = self._transcribe_paths(window)
                transcribe_time_total += time.time() - t0
                collected_outputs.extend(outputs)
                # Diarization removed

        tail_start_index = (len(chunk_paths) // self.WINDOW_SIZE) * self.WINDOW_SIZE
        tail_window = chunk_paths[tail_start_index:]
        if tail_window:
            t0 = time.time()
            outputs = self._transcribe_paths(tail_window)
            transcribe_time_total += time.time() - t0
            collected_outputs.extend(outputs)
            # Diarization removed

        return (
            collected_outputs,
            collected_segments,
            transcribe_time_total,
            diar_time_total,
        )

    # Helper: cleanup temp chunk dir and input wav
    def _cleanup_temp(self, chunk_dir: str, tmp_input_wav: str):
        for fn in os.listdir(chunk_dir):
            try:
                os.remove(os.path.join(chunk_dir, fn))
            except Exception:
                pass
        try:
            os.rmdir(chunk_dir)
        except Exception:
            pass
        try:
            if os.path.exists(tmp_input_wav):
                os.remove(tmp_input_wav)
        except Exception:
            pass

    def _transcribe_paths(self, paths):
        if not paths:
            return []
        with torch.autocast(
            "cuda", enabled=True, dtype=self.COMPUTE_TYPE
        ), torch.inference_mode(), torch.no_grad():
            outputs = self.asr_model.transcribe(
                paths,
                batch_size=self.TRANSCRIBE_BATCH_SIZE,
                num_workers=self.TRANSCRIBE_WORKERS,
                timestamps=True,
            )
        if isinstance(outputs, tuple) and len(outputs) == 2:
            outputs = outputs[0]
        # Return full outputs so we can use timestamps downstream
        return outputs

    @method()
    async def transcribe(
        self,
        uid: str,
        audio_file_path: str,
        enable_speakers: bool = False,
        num_speakers: int = 1,
    ):
        audio_storage_vol.reload()
        # Diarization removed; proceed with ASR-only regardless of enable_speakers

        start_time = time.time()
        chunk_dir = None

        sample_rate = self.SAMPLE_RATE
        chunk_duration_sec = self.CHUNK_SECONDS
        overlap_sec = self.OVERLAP_SECONDS

        import tempfile

        base_dir = os.path.dirname(audio_file_path) or "."
        tmp_parent = (
            "/dev/shm"
            if os.path.isdir("/dev/shm")
            else ("/tmp" if os.path.isdir("/tmp") else base_dir)
        )
        chunk_dir = tempfile.mkdtemp(prefix="chunks_", dir=tmp_parent)
        logger.info(f"📂 Using temp dir: {chunk_dir}")

        # Convert to 16k mono WAV in RAM and process sequentially per window
        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        tmp_input_wav = self._convert_to_wav(
            audio_file_path, sample_rate, tmp_parent, base_name
        )

        collected_outputs = []  # NeMo outputs with text and timestamp
        collected_segments = []
        chunk_paths, io_time_total = self._produce_chunks(
            tmp_input_wav,
            sample_rate,
            chunk_duration_sec,
            overlap_sec,
            chunk_dir,
        )
        produce_t0 = time.time()
        (
            collected_outputs,
            collected_segments,
            transcribe_time_total,
            diar_time_total,
        ) = self._process_windows(chunk_paths, enable_speakers)
        produce_t1 = time.time()

        # Build transcript from collected outputs
        transcript = " ".join(
            [getattr(o, "text", "") for o in collected_outputs]
        ).strip()
        logger.info(
            f"⏱️ Producer {produce_t1 - produce_t0:.2f}s (I/O {io_time_total:.2f}s), "
            f"ASR_total {transcribe_time_total:.2f}s, DIAR_total {diar_time_total:.2f}s, chunks {len(chunk_paths)}"
        )

        # Cleanup
        self._cleanup_temp(chunk_dir, tmp_input_wav)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Speaker diarization not available in Parakeet path
        speaker_transcript = None

        # with Session(engine, expire_on_commit=False) as session:
        #     transcriptObj = find_transcript_by_uid(UUID(uid), session)
        total_time = time.time() - start_time
        output_data = {
            "text": transcript,
            "processing_time": total_time,
            "language": "en",
            "speaker_transcript": speaker_transcript,
        }

        # transcriptObj.transcript = output_data["text"]
        # transcriptObj.language = output_data["language"]
        # update_or_create_episodes(
        #     transcriptObj,
        #     include_fields=["transcript", "language"],
        #     exclude_fields=["uid"],
        #     upsert=False,
        # )
        # Force container shutdown after returning result
        # os._exit(0) # SIGKILL the process
        # sys.exit(0)  # SIGTERM the process
        return output_data, total_time

    @modal_exit()
    def close_container(self):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Shutting down Parakeet container")

from modal import Volume

audio_storage_vol = Volume.from_name("audio_storage_vol", create_if_missing=True)
transcriptions_vol = Volume.from_name("transcriptions_vol", create_if_missing=True)

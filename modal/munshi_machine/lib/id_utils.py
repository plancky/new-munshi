def generate_file_id_from_hash(content_hash: str, source_type: str = "local") -> str:
    if source_type == "podcast":
        return f"podcast_{content_hash}"
    return f"local_{content_hash}"


def get_hash_from_file_id(file_id: str) -> str:
    if file_id.startswith("podcast_"):
        return file_id.replace("podcast_", "")
    if file_id.startswith("local_"):
        return file_id.replace("local_", "")
    return file_id

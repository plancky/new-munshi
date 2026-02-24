from pydantic import BaseModel

class GeminiResponse(BaseModel):
    cleaned_text: list[str]

class GeminiSpeakerResponse(BaseModel):
    cleaned_transcript: list[str]
    speaker_ids: list[str]
    speaker_names: list[str]

class RagAnswerResponse(BaseModel):
    answer_html: str
    answer_title: str
    found_answer: bool

class Insight(BaseModel):
    text: str

class ComprehensiveSummaryResponse(BaseModel):
    summary: str
    insights: list[Insight]
    tangents: list[Insight]


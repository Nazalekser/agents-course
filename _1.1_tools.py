from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class SentenceEntities(BaseModel):
    entities: list[str]


def response(instructions: str, user_input: str, format: BaseModel):
    response = client.responses.parse(
        model="gpt-4.1",
        instructions=instructions,
        input=user_input,
        format=SentenceEntities,
    )
    return response.output_parsed


def process_transcript(transcript: str):
    summary = response("Summarize into 3-5 sentences", transcript)
    tone = response("Determine the tone of the transcript", transcript)
    return summary, tone


def main():
    transcript = "Hello, how are you?"
    summary, tone = process_transcript(transcript)
    print(summary)
    print(tone)


if __name__ == "__main__":
    main()

import json
from openai import OpenAI
from utils import fn_to_schema, tag


class MiniAgent:

    def __init__(self, instructions: str, tools: list, model: str = "o4-mini"):
        self.client = OpenAI()
        self.instructions, self.model = instructions, model
        self.tools = {fn.__name__: fn for fn in tools}
        self.tools_schema = [fn_to_schema(fn) for fn in tools]

    # ---------- item handler -------------------------------------------
    def _handle_item(self, item):

        if item.type == "reasoning":
            print(tag("reasoning") + "".join(item.summary))
            return []

        if item.type == "message":
            txt = "".join(p.text for p in item.content if p.type == "output_text")
            print(tag("message") + txt)
            return []

        if item.type == "function_call":
            args = json.loads(item.arguments or "{}")
            print(tag("function_call") + f"{item.name}({json.dumps(args)})")
            result = self.tools[item.name](**args) if args else self.tools[item.name]()
            print(tag("function_output") + json.dumps(result))
            return [
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result),
                }
            ]

        return []

    # ---------- main loop ----------------------------------------------
    def run(self, user_text: str):
        print("Input:", user_text)
        turn_input = [{"role": "user", "content": user_text}]
        prev_id = None

        while turn_input:
            stream = self.client.responses.create(
                model=self.model,
                instructions=self.instructions,
                tools=self.tools_schema,
                input=turn_input,
                previous_response_id=prev_id,
                stream=True,
            )
            turn_input = []  # collect next‑turn inputs

            for event in stream:
                if event.type == "response.output_item.done":
                    turn_input += self._handle_item(event.item)

                if event.type == "response.completed":
                    prev_id = event.response.id

        return prev_id


def add(a: str, b: str) -> int:
    """Add two numbers together and return the result."""
    return int(a) + int(b)


def send_email(to: str, subject: str, body: str):
    """Send an email to the given address with the given subject and body."""
    print(f"Sending email to {to} with subject {subject} and body {body}")


if __name__ == "__main__":
    tools = [add, send_email]
    agent = MiniAgent(
        model="gpt-4.1",
        instructions="You are a helpful assistant that can add numbers. Call the `add` tool to add numbers.",
        tools=tools,
    )
    agent.run(
        "Send an email to John Doe with the subject 'Hello' and body 'How are you?'"
    )

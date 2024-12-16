class PromptService:
    """
    Service to manage versioned prompts for the OpenAI API.
    """

    def __init__(self):
        self.prompts = {
            "v1": """
Act as a professional writer with extensive expertise in dermatology. 
Your task is to enhance the readability and flow of a letter without subheadings, 
approximately 500 words long, ensuring it sounds like it was written by a native speaker. 
While improving the writing style, maintain the original meaning and intent of the text. 
Here is the letter:
{text}
"""
            # Future versions can be added here
        }

    def get_prompt(self, version: str, text: str) -> str:
        if version not in self.prompts:
            raise ValueError(f"Prompt version '{version}' not found.")

        return self.prompts[version].format(text=text)

    def add_prompt(self, version: str, prompt_template: str) -> None:
        self.prompts[version] = prompt_template

    def list_versions(self) -> list:
        return list(self.prompts.keys())

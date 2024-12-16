class PolishService:
    """
    Service to handle polishing a single article using the OpenAI API.
    """

    def __init__(self, client, prompt_service, prompt_version: str, model: str = "chatgpt-4o-latest", temperature: float = 0.7):
        self.client = client
        self.prompt_service = prompt_service
        self.prompt_version = prompt_version
        self.model = model
        self.temperature = temperature

    def polish_article(self, article: str) -> str:
        if not article:
            raise ValueError("The article text is empty and cannot be polished.")

        # Get the formatted prompt from the PromptService
        prompt = self.prompt_service.get_prompt(self.prompt_version, article)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )

            # Extract the polished text from the response
            polished_text = response.choices[0].message.content
            return polished_text

        except Exception as e:
            raise ValueError(f"OpenAI API call failed: {e}")

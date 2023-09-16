import openai

def promptEngineer(gender, attire_desc, occasion_desc):
    prompt_to_gpt = f"I am a {gender}, wearing a {attire_desc}. Details of the occasion are as follows: {occasion_desc}. Give one overall rating out of 'consider changing outfit', 'could be better', 'on point', and title it 'overall rating'. Consider: Color coordination,  Color appropriateness, Potential cultural sensitivity, Seasonal suitability, accessories (for each of the categories give one succinct sentence."

    return prompt_to_gpt

def gpt_inference(prompt_to_gpt):
    prompt_to_gpt = get_prompts(prompt_to_gpt)
    print(f"{prompt_to_gpt = }")
    response = generate_chat_completion(prompt_to_gpt)

    return response

def generate_chat_completion(
    messages: dict[str, str | list[dict[str, str]]], temperature: int = 1, max_tokens: int | None = None
) -> dict[str, str] | None:
    openai.api_key = "<API_Key>"
    response_text = openai.ChatCompletion.create(
                model=messages["model"], messages=messages["messages"], temperature=temperature, max_tokens=max_tokens
            )

    response = response_text['choices'][0]['message']['content']

    return response

def get_prompts(prompt_to_gpt) -> dict[str, str | list[dict[str, str]]]:
    """Get prompts for the given style"""
    gpt_model = "gpt-3.5-turbo"
    message = {
        'model': gpt_model,
        'messages': [
            {
                "role": "system",
                "content": "You are a helpful assistant who helps me rate my attire's appropriate-ness for an event or occasion out of 10.",
            },
            {
                "role": "user",
                "content": "Please rate my attire's appropriate-ness for an event or occasion out of 10. Please have a strong opinion about it."
            },
            {"role": "user", "content": prompt_to_gpt},
        ],
    }

    return message

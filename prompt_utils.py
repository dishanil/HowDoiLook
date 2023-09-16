import openai

def promptEngineer(attire_desc, occasion_desc):
    prompt_to_gpt = "My clothing attire consists of " + attire_desc.strip() + " and the place or event I am going for can be described as " + occasion_desc.strip() + "." + "Do you think I am appropriately dressed? Please rank it out of 10 where 10 is the most appropriate."
    return prompt_to_gpt

def gpt_inference(prompt_to_gpt):
    prompt_to_gpt = get_prompts(prompt_to_gpt)
    print(f"{prompt_to_gpt = }")
    response = generate_chat_completion(prompt_to_gpt)

    return response

def generate_chat_completion(
    messages: dict[str, str | list[dict[str, str]]], temperature: int = 1, max_tokens: int | None = None
) -> dict[str, str] | None:
    openai.api_key = "<use API key>"
    response_text = openai.ChatCompletion.create(
                model=messages["model"], messages=messages["messages"], temperature=temperature, max_tokens=max_tokens
            )

    response = response_text['choices'][0]['message']['content']

    return response

def getSuggestions(occasion_desc):
    prompt = occasion_desc.strip() + " What should I wear to this?"

    return "Suggestion: \n Wear anything please!"

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
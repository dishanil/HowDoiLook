def promptEngineer(attire_desc, occasion_desc):
    prompt_to_gpt = "My clothing attire consists of " + attire_desc.strip() + " and the place or event I am going for can be described as " + occasion_desc.strip()
    return prompt_to_gpt

def gpt_inference(prompt_to_gpt):
    return "You look amazing"

def getSuggestions(occasion_desc):
    prompt = occasion_desc.strip() + " What should I wear to this?"

    return "Suggestion: \n Wear anything please!"

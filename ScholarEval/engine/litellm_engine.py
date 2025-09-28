import openai
class LLMEngine():
    def __init__(self, llm_engine_name, api_key, api_endpoint):
        self.llm_engine_name = llm_engine_name
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.client = openai.OpenAI(api_key=api_key, base_url=api_endpoint) 
        

    def respond(self, user_input, temperature = 0.7, top_p = 0.95, max_tokens = 40000):
        response = self.client.chat.completions.create(model=self.llm_engine_name,messages=user_input, temperature=temperature)
        
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens

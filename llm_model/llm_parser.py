import  json, re
from config import OPENAI_API_KEY
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

def parse_card_with_llm(text):
    text = text.strip()
    text = text.encode("utf-8", "ignore").decode("utf-8")

    prompt = f"""
    아래 명함 텍스트에서 가능한 항목(name, company, position, email, phone, address)을 json으로 반환.
    {text}
    """

    try:
        res = client.chat.completions.create(
            model = "gpt-4o",
            messages = [{"role": "user", "content": prompt}],
            temperature = 0
        )
        content = res.choices[0].message.content
        print("LLM 응답:", content)

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            json_text = match.group(0)
        else:
            json_text = "{}"

        data =json.loads(json_text)
        return data
    except Exception as e:
        print("LLM failed",e)
        return {}


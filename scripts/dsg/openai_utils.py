
import openai
import os


base_url = os.environ.get('OPENAI_API_URL')
key = os.environ.get('ARK_API_KEY')
api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-03-01-preview")


def openai_setup(key_path='./_OAI_KEY.txt'):
	with open(key_path) as f:
		key = f.read().strip()

	print("Read key from", key_path)
	openai.api_key = key

def openai_completion(
	prompt,
	model='gpt-3.5-turbo-16k-0613',
	temperature=0,
	return_response=False,
	max_tokens=500,
	):

	if 'gpt' in model:
		client = openai.AzureOpenAI(
			azure_endpoint=base_url,
			api_version=api_version,
			api_key=key,
		)

	else:
		client = openai.OpenAI(
			base_url=base_url,
			api_key=key,
		)

	resp = client.chat.completions.create(
		model=model,
		messages=[{"role": "user", "content": prompt}],
		temperature=temperature,
		max_tokens=max_tokens,
		extra_headers={"X-TT-LOGID": ""},
		timeout=1800000
	)
	
	if return_response:
		return resp

	return resp.choices[0].message.content
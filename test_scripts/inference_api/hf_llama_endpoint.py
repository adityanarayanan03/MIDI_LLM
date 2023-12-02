import requests

API_URL = "https://e3mbnwdg9emmw758.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Authorization": "Bearer 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",
	"Content-Type": "application/json"
}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "A AB ABC ABCD ABCDE ABCDEF",
})

print(output)
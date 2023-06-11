import os

from gpru.azure.preview_2023_06_01 import AzureOpenAiApi, ImageRequest, ImageSize

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, key)

req = ImageRequest(prompt="An avocado chair", size=ImageSize.SQUARE_512, n=3)
image_operation = api.generate_images(req)
print(image_operation.json(indent=2))
# Example output:
# {
#   "status": "notRunning",
#   "id": "f508bcf2-e651-4b4b-85a7-58ad77981ffa"
# }

: "${ALIBABA_API_KEY:?ALIBABA_API_KEY is not set. Load .envrc or export it first.}"

IMG=$(base64 -i ~/Pictures/miofa.jpg | tr -d '\n') \
curl https://coding-intl.dashscope.aliyuncs.com/v1/chat/completions \
  -H "Authorization: Bearer ${ALIBABA_API_KEY}" \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "qwen3.5-plus",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Which animal does this look like?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,$IMG"
          }
        }
      ]
    }
  ],
  "temperature": 0.2
}
EOF

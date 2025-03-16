# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import os

from openai import OpenAI
from transformers.utils.versions import require_version


require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")


def main():
    client = OpenAI(
        api_key="{}".format(os.environ.get("API_KEY", "0")),
        # base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000)),
        base_url="http://localhost:8000/v1",
    )
    messages = []
    image_path = "/workspace/llama_factory/app/mine/data/llava-zh-3k-data/1_1.jpg"
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{encoded_string}"
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述图像。"},
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
            ],
        }
    )
    result = client.chat.completions.create(messages=messages, model="/workspace/modelscope/Qwen/Qwen2.5-VL-72B-Instruct-AWQ")  # Qwen/Qwen2-VL-7B-Instruct
    messages.append(result.choices[0].message)
    print("Round 1:", result.choices[0].message.content)
    # The image shows a pyramid of colored blocks with numbers on them. Here are the colors and numbers of ...
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "这是什么花?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/flowers.jpg"},
            },
        ],
    }]
    result = client.chat.completions.create(messages=messages, model="/workspace/modelscope/Qwen/Qwen2.5-VL-72B-Instruct-AWQ")
    messages.append(result.choices[0].message)
    print("Round 2:", result.choices[0].message.content)
    # The image shows a cluster of forget-me-not flowers. Forget-me-nots are small ...
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这种花生长在哪些地方？"}
            ],
        }
    )
    result = client.chat.completions.create(messages=messages,
                                            model="/workspace/modelscope/Qwen/Qwen2.5-VL-72B-Instruct-AWQ")
    messages.append(result.choices[0].message)
    print("Round 3:", result.choices[0].message.content)


if __name__ == "__main__":
    main()
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
        base_url="http://localhost:9000/v1",
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "你是谁？"}
        ],
    }]
    result = client.chat.completions.create(messages=messages, model="test")  # Qwen/Qwen2-VL-7B-Instruct
    messages.append(result.choices[0].message)
    print("Round 1:", result.choices[0].message.content)
    # The image shows a pyramid of colored blocks with numbers on them. Here are the colors and numbers of ...
    image_path = "/workspace/llama_factory/app/mine/data/dimao_vision_human_data/1113-0010-0001-0003-0216.png"
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    data_url = f"data:image/png;base64,{encoded_string}"
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图像"},
            {
                "type": "image_url",
                "image_url": {"url": image_path},
            },
        ],
    }]
    result = client.chat.completions.create(messages=messages, model="test")
    messages.append(result.choices[0].message)
    print("Round 2:", result.choices[0].message.content)
    # The image shows a cluster of forget-me-not flowers. Forget-me-nots are small ...
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图像在表达什么地貌学原理？"}
            ],
        }
    )
    result = client.chat.completions.create(messages=messages, model="test")
    messages.append(result.choices[0].message)
    print("Round 3:", result.choices[0].message.content)


if __name__ == "__main__":
    main()
import requests

# 定义请求的URL
url = 'http://localhost:11434/api/tags'

# 发送GET请求
try:
    response = requests.get(url)

    # 检查响应状态码
    if response.status_code == 200:
        # 输出响应内容
        print('Response content:', response.text)
    else:
        # 输出错误信息
        print('Failed to retrieve data. Status code:', response.status_code)
except requests.exceptions.RequestException as e:
    # 捕获并输出请求异常
    print('An error occurred:', e)
import torch

if __name__ == '__main__':
    print(torch.__version__)          # 应显示2.5.1+cu121或自定义版本
    print(torch.version.cuda)          # 应显示12.1或12.3（取决于方案）
    print(torch.cuda.is_available())  # 应为True
import os
import io
import base64
import json
from typing import Dict, Any
from openai import OpenAI
from PIL import Image
import httpx


class QwenAPIClient:
    def __init__(self, api_key: str):
        """
        初始化Qwen API客户端

        Args:
            api_key: Qwen API密钥
        """
        self.api_key = api_key
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        # 创建httpx客户端以支持代理
        http_client = httpx.Client()

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=http_client
        )


    def encode_image_to_base64(self, image_path: str) -> str:
        """
        将图片编码为base64格式

        Args:
            image_path: 图片路径

        Returns:
            base64编码的图片字符串
        """
        with Image.open(image_path) as img:
            # 将图片转换为RGB格式（如果是RGBA的话）
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            # 将图片保存到字节流
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")

            # 编码为base64
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return img_base64

    def create_prompt(self) -> str:
        """
        从文件加载提示词
        """
        prompt_file = os.path.join(os.path.dirname(__file__), "prompts", "driving_scene_analysis.txt")
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt文件不存在: {prompt_file}")
        except Exception as e:
            raise Exception(f"读取Prompt文件时出错: {e}")

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        分析图片并返回结构化的JSON结果

        Args:
            image_path: 图片路径

        Returns:
            包含场景和物体检测结果的字典
        """
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        # 编码图片
        image_base64 = self.encode_image_to_base64(image_path)

        # 获取提示词
        prompt = self.create_prompt()

        completion = self.client.chat.completions.create(
            model="qwen-vl-max-latest", # 此处以qwen-vl-max-latest为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
            messages=[
                {
                    "role": "system",
                    "content": "你是一个驾驶场景标注专家，严格按照要求输出JSON格式结果。"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )

        # 解析返回结果
        try:
            result_text = completion.choices[0].message.content
            # 尝试解析JSON
            result = json.loads(result_text)
            return result
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始返回: {result_text}")
            return {"error": "JSON解析失败", "raw_response": result_text}
        except Exception as e:
            print(f"处理错误: {e}")
            return {"error": str(e)}


def main():
    """
    主函数 - 示例用法
    """
    # 配置参数
    API_KEY = os.environ.get('QWEN_API_KEY', '')
    IMAGE_PATH = "data/test.jpg"  # 图片路径

    # 检查API密钥
    if not API_KEY:
        print("错误：未找到QWEN_API_KEY环境变量！")
        print("\n请按以下步骤设置API密钥：")
        print("1. 访问 https://dashscope.aliyuncs.com/")
        print("2. 注册/登录阿里云账号")
        print("3. 开通DashScope服务")
        print("4. 获取API-KEY")
        print("5. 在 ~/.bashrc 中添加：export QWEN_API_KEY=your_key_here")
        print("6. 执行：source ~/.bashrc")
        return

    # 检查图片文件
    if not os.path.exists(IMAGE_PATH):
        print(f"图片文件不存在: {IMAGE_PATH}")
        print("请确保图片文件存在于当前目录")
        return

    print(f"分析图片: {IMAGE_PATH}")

    # 创建客户端
    client = QwenAPIClient(api_key=API_KEY)

    # 分析图片
    print("正在分析图片...")
    result = client.analyze_image(IMAGE_PATH)

    # 输出结果
    if "error" in result:
        print(f"分析失败: {result['error']}")
    else:
        print("分析完成！")
        print("\n分析结果：")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
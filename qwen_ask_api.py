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
        创建与原版相同的提示词
        """
        return (
            "你是一位驾驶场景标注专家。\n\n"
            "输入：\n"
            "1. 输入图片是车载相机拍摄的一张图片。\n\n"
            "任务：\n"
            "1. 检查你当前所处的地方是否是以下位置：\n"
            "- 路口\n"
            "- 直路\n"
            "- 匝道\n"
            "- 弯道\n\n"
            "2. 检查图片中是否存在以下物体。车辆（vehicle）可以是汽车、公交车、卡车、摩托车手、滑板车等。交通元素（traffic_element）包括交通标志和交通信号灯。道路危险（road_hazard）可能包括危险道路条件、道路碎片、障碍物等。冲突车辆（conflicting_vehicle）是指可能与自车未来路径发生潜在冲突的车辆。需审计的对象类别：\n"
            "- nearby_vehicle (附近车辆)\n"
            "- pedestrian (行人)\n"
            "- cyclist (骑行者)\n"
            "- construction (施工)\n"
            "- traffic_element (交通元素)\n"
            "- weather_condition (天气条件)\n"
            "- road_hazard (道路危险)\n"
            "- emergency_vehicle (紧急车辆)\n"
            "- animal (动物)\n"
            "- special_vehicle (特殊车辆)\n"
            "- conflicting_vehicle (冲突车辆)\n"
            "- door_opening_vehicle (开门车辆)\n"
            "3. 对每个类别输出\"yes\"或\"no\"（不能遗漏）。\n\n"
            "输出格式（严格的JSON，无额外键，无注释）：\n\n"
            "{\n"
            "\"scenarios\": {\n"
            "\"junction\": \"yes | no\",\n"
            "\"straight_road\": \"yes | no\",\n"
            "\"ramp_entrance\": \"yes | no\",\n"
            "\"ramp_exit\": \"yes | no\",\n"
            "\"curve\": \"yes | no\"\n"
            "},\n"
            "\"critical_objects\": {\n"
            "\"nearby_vehicle\": \"yes | no\",\n"
            "\"pedestrian\": \"yes | no\",\n"
            "\"cyclist\": \"yes | no\",\n"
            "\"construction\": \"yes | no\",\n"
            "\"traffic_element\": \"yes | no\",\n"
            "\"weather_condition\": \"yes | no\",\n"
            "\"road_hazard\": \"yes | no\",\n"
            "\"emergency_vehicle\": \"yes | no\",\n"
            "\"animal\": \"yes | no\",\n"
            "\"special_vehicle\": \"yes | no\",\n"
            "\"conflicting_vehicle\": \"yes | no\",\n"
            "\"door_opening_vehicle\": \"yes | no\"\n"
            "}\n"
            "}"
        )

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
import os
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = "/root/models/Qwen2.5-VL-7B-Instruct/" #"Qwen/Qwen2.5-VL-7B-Instruct"
assert os.path.exists(model_path)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

#模型中每幅图像的视觉标记数量的默认范围为 4-16384。
#您可以根据需要设置 min_pixels 和 max_pixels，例如 256-1280 的 token 范围，以平衡性能和显存占用
processor = AutoProcessor.from_pretrained(
    model_path,
    # min_pixels=256*28*28, max_pixels=1280*28*28,
)

print('\n模型加载完毕！')

img_path = "/root/data/test.jpg"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img_path,  # 请确保 img_path 已正确定义
            },
            {
                "type": "text", 
                "text": (
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
                    "3. 对每个类别输出“yes”或“no”（不能遗漏）。\n\n"
                    "输出格式（严格的JSON，无额外键，无注释）：\n\n"
                    "{\n"
                    "\"scenarios\": {\n"
                    "\"junction\": \"yes | no\",\n"
                    "\"straight_road\": \"yes | no\",\n"  # 修正键名
                    "\"ramp_entrance\": \"yes | no\",\n"  # 修正键名
                    "\"ramp_exit\": \"yes | no\",\n"      # 修正键名
                    "\"curve\": \"yes | no\"\n"           # 修正键名，移除重复项
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
                    "\"door_opening_vehicle\": \"yes | no\"\n"  # 移除多余引号
                    "}\n"  # 添加缺失的闭合括号
                    "}"
                )
            }
        ]
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print('\n回答：\n', output_text)
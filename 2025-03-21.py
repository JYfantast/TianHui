from datetime import datetime
import pandas as pd
import argparse
import logging
import json
import os

#自定义
from model.API import response_API
from model.Qwen2_5 import model_tokenizer_Qwen, response_Qwen
from model.baichuan2 import model_tokenizer_baichuan2, model_tokenizer_lora_baichuan2, response_baichuan2
from model.chatglm3 import model_tokenizer_chatglm3, response_chatglm3
from model.Llama3 import model_tokenizer_Llama3,response_Llama3
from model.bianque2 import model_tokenizer_bianque2,response_bianque2
from model.LLaMA7B import model_tokenizer_lora_LLaMA7B, response_LLaMA7B
from model.Lingdan import model_tokenizer_Lingdan, response_Lingdan

logger = logging.getLogger(__name__)

#需传入参数
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default="Qwen2.5-Math-7B",
                    help='"DeepSeek-R1-Distill-Qwen-7B","DeepSeek-R1-Distill-Qwen-14B","DeepSeek-R1-Distill-Qwen-32B",'
                         '"Qwen2.5-Math-7B","Qwen2.5-14B-Instruct","Qwen2.5-32B-Instruct","Qwen2.5-72B-Instruct","Baichuan-M1-14B-Instruct",'
                         '"Baichuan2-7B-Chat","Baichuan2-13B-Chat","chatglm3-6b","Llama3-8B-Chinese-Chat",'
                         '"BianQue-2","chatmed",'
                         '"Lingdan-13B-Base","Lingdan-13B-PR","HuatuoGPT2-13B","Sunsimiao-7B",'
                         '"BianCang-Qwen2.5-7B-Instruct","TCMchat","ZhongjingGPT1_13B","Bentao","ShenNong",'
                         '"gpt-3.5-turbo","gpt-4o","o1","deepseek-r1","deepseek-v3",'
                         '"our_model-7B","our_model-14B","our_model-32B",'
                         '"14B.128.256.0.2.4","14B.64.128.0.2.4","14B.32.64.0.2.4","14B.16.32.0.2.4","14B.8.16.0.2.4",'
                         '"14B.128.256.0.0.4","14B.128.256.0.2.2","14B.128.256.0.2.6","14B.128.256.0.4.4",'
                         '"14B.128.256.0.2.4.1024","14B.128.256.0.2.4.256","14B.128.256.0.2.4.512"')

parser.add_argument('--question_type', type=str, default="QA",
                    help='"choice","QA","RC","NER","MC","recommend","instructions",'
                         '"chemical","pharmacological","abstract","title","acupoint"')

args = parser.parse_args()

# 解析命令行参数
model_type = args.model_type
question_type = args.question_type

# 获取当前时间并格式化
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

#构建prompt文件路径
if question_type == "choice":
    demo_file_path = ("问题："
                      "大黄具有的功效是____"
                      "A：泻下，清热，软坚"
                      "B：泻下，清热，利尿"
                      "C：泻下，清热，杀虫"
                      "D：泻下，清热，止血"
                      "E：泻下，清热，祛痰"
                      "回答："
                      "D"
                      "以上是示例。请学习以上回答，只有一个选项是正确选项，只输出正确选项，比如A、B、C、D、E，不要输出解释和思考。")
    input_file_path = "/data/1JY/5TCMChat/evaluation/all/test_data_choice.csv"
    # 检查并创建文件夹
    if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
        os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
    output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/choice.{current_time}.result.csv"
# if question_type == "choice":
#     demo_file_path = ("")
#     input_file_path = "/data/1JY/5TCMChat/evaluation/all/test_data_choice.csv"
#     # 检查并创建文件夹
#     if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
#         os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
#     output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/choice.{current_time}.result.csv"
elif question_type == "recommend":
    demo_file_path = ("问题："
                      "以下是关于中医药从相似组方化学成分谱方面来推荐方剂的任务。"
                      "成药强筋健骨胶囊与哪些中成药具有相似组方化学成分谱的中成药？"
                      "回答："
                      "强筋健骨片、强筋健骨丸、健脾颗粒、健脾丸（浓缩丸）、强肾镇痛丸"
                      "以上是示例。请学习以上回答，对以下问题进行回答。")
    input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.recommend.json"
    # 检查并创建文件夹
    if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
        os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
    output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/recommend.{current_time}.result.csv"
# elif question_type == "recommend":
#     demo_file_path = ("")
#     input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.recommend.json"
#     # 检查并创建文件夹
#     if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
#         os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
#     output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/recommend.{current_time}.result.csv"
elif question_type == "RC":
    demo_file_path = ("问题："
                      "以下是关于中医药文本阅读理解任务。"
                      "据新华社电美国内布拉斯加医疗中心17日说，因感染埃博拉病毒在该医院接受治疗的塞拉利昂医生马丁·萨利亚已经病重去世。此前，美国已收治9名埃博拉病毒感染者，其中8人痊愈，他们全是美国公民。请问美国已收治几名埃博拉病毒感染者，其中8人痊愈？"
                      "回答："
                      "美国已收治9名埃博拉病毒感染者，其中8人痊愈，"
                      "以上是示例。请学习以上回答，对以下问题进行回答。")
    input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.RC.json"
    # 检查并创建文件夹
    if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
        os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
    output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/RC.{current_time}.result.csv"
# elif question_type == "RC":
#     demo_file_path = ("")
#     input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.RC.json"
#     # 检查并创建文件夹
#     if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
#         os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
#     output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/RC.{current_time}.result.csv"
elif question_type == "QA":
    demo_file_path = ("问题："
                      "请介绍云门，包括所在经络，定位，解剖位置，功效。"
                      "回答："
                      "云门是手太阴肺经的重要腧穴，位于胸部的锁骨下窝凹陷中，具体定位是在肩胛骨喙突内缘与前正中线之间，旁开约6寸的位置。解剖位置上，云门穴的周围有胸大肌覆盖。这个穴位布有胸前神经的分支、臂丛外侧束以及锁骨上神经的中后支。在功效方面，云门穴常用于治疗咳嗽、气喘以及胸痛等与胸肺相关的病症，同时也能缓解肩背部的疼痛。"
                      "以上是示例。请学习以上回答，对以下问题进行回答。")
    input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.QA.json"
    # 检查并创建文件夹
    if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
        os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
    output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/QA.{current_time}.result.csv"
# elif question_type == "QA":
#     demo_file_path = ("")
#     input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.QA.json"
#     # 检查并创建文件夹
#     if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
#         os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
#     output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/QA.{current_time}.result.csv"
# elif question_type == "pharmacological":
#     demo_file_path = ("问题："
#                       "益母草的药理作用有哪些？"
#                       "回答："
#                       "益母草是一种传统中药，其药理作用十分显著。研究表明，益母草的煎剂、乙醇浸膏及益母草碱均具有兴奋子宫的效果，对小鼠实验也显示出一定的抗着床和抗早孕的作用。此外，益母草注射液能够有效保护心肌，减轻缺血再灌注损伤，并具备抗血小板聚集和降低血液黏度的能力。益母草的粗提物能够扩张血管，表现出短暂的降压效果，而益母草碱则具有明显的利尿作用。这些药理特性使得益母草在临床上被广泛应用于多种疾病的治疗。"
#                       "以上是示例。请学习以上回答，对以下问题进行回答。")
#     input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.pharmacological.json"
#     # 检查并创建文件夹
#     if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
#         os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
#     output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/pharmacological.{current_time}.result.csv"
elif question_type == "pharmacological":
    demo_file_path = ("问题：益母草的药理作用有哪些？回答：益母草是一种传统中药，其药理作用十分显著。研究表明，益母草的煎剂、乙醇浸膏及益母草碱均具有兴奋子宫的效果，对小鼠实验也显示出一定的抗着床和抗早孕的作用。此外，益母草注射液能够有效保护心肌，减轻缺血再灌注损伤，并具备抗血小板聚集和降低血液黏度的能力。益母草的粗提物能够扩张血管，表现出短暂的降压效果，而益母草碱则具有明显的利尿作用。这些药理特性使得益母草在临床上被广泛应用于多种疾病的治疗。"
                      "问题：棕榈炭的药理作用有哪些？回答：棕榈炭具有多种药理作用，其中最为显著的是其止血作用。研究表明，无论是使用陈棕皮炭、陈棕炭，还是其水煎剂和混悬剂进行灌胃给药，都能有效缩短小鼠的出血和凝血时间。这一特性使得棕榈炭在医学领域中，尤其是在止血和促进创伤愈合方面，展现出一定的应用潜力。"
                      "问题：自然铜的药理作用有哪些？回答：自然铜是一种具有多种药理作用的矿物元素。研究表明，自然铜能有效促进骨折愈合，其机制表现为加速骨痂的生长，使其在数量和成熟度上均有显著提升。此外，自然铜对多种病原性真菌也展现出一定的拮抗作用，能在不同程度上抑制这些病原菌的生长。这些药理特性使得自然铜在医学领域中的应用前景广阔。"
                      "以上是示例。请学习以上回答，对以下问题进行回答。")
    input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.pharmacological.json"
    # 检查并创建文件夹
    if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
        os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
    output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/pharmacological.{current_time}.result.csv"
# elif question_type == "pharmacological":
#     demo_file_path = ("")
#     input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.pharmacological.json"
#     # 检查并创建文件夹
#     if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
#         os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
#     output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/pharmacological.{current_time}.result.csv"
elif question_type == "NER":
    demo_file_path = ("问题："
                      "以下是关于中成药说明书实体识别任务。"
                      "【药品商品名称】 乌鸡白凤丸 【药品名称】 乌鸡白凤丸【批准文号】 国药准字Z21021637 【成分】 乌鸡(去毛、爪、肠)、鹿角胶、当归、白芍、熟地黄、人参、黄芪、香附(醋制)、丹参、桑螵蛸、鹿角霜、牡蛎(煅)等味。 【功效】补气养血、调经止带，用于月经不调、经期腹痛【用法用量】 口服。一次9克，一日1次；或将药丸加适量开水溶后服。 本药内所含人参、白芍，反藜芦，忌与含藜芦的药物同用。 本药内所含甘草，反甘遂、大戟、海藻、芫花，忌与含甘遂、大戟、海藻、芫花的药物同用。服药期间避免与生冷、辛辣、荤腥油腻、不易消化食品同用，戒烟酒。服药期间不宜喝茶和吃萝卜，不宜同时服用五灵脂、皂荚或其制剂。医师和药师可能对服用同仁乌鸡白凤丸萝卜，不宜同时服用五灵脂、皂荚或其制剂。"
                      "回答："
                      "症状:月经不调，经期腹痛；人群:孕妇；食物:萝卜；药物成分:乌鸡，鹿角胶，当归，白芍，熟地黄，人参，黄芪，香附，丹参，桑螵蛸，鹿角霜，牡蛎，藜芦，甘草，甘遂，大戟，海藻，芫花，五灵脂，皂荚；中药功效:补气养血，调经止带，促进造血，止血；食物分组:生冷，辛辣，荤腥，油腻，不易消化；药物:乌鸡白凤丸。"
                      "以上是示例。请学习以上回答，对以下问题进行回答。")
    input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.NER.json"
    # 检查并创建文件夹
    if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
        os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
    output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/NER.{current_time}.result.csv"
# elif question_type == "NER":
#     demo_file_path = ("")
#     input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.NER.json"
#     # 检查并创建文件夹
#     if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
#         os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
#     output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/NER.{current_time}.result.csv"
elif question_type == "MC":
    demo_file_path = ("问题："
                      "根据病史、症状等医案信息，对患者进行证候、疾病诊断，并推荐相关的中药或者方剂。"
                      "患者病史患者自诉5年前开始于双腋窝部出现一种特殊的刺鼻的臭味，夏季出汗时更甚，且逐年明显，患者为求彻底祛除臭味，遂今日来我科住院手术治疗。临床症状神志清晰，精神尚可，形体适中，语言清晰，口唇红润；皮肤同前，有斑疹。胸部对称，虚里搏动正常，腹部平坦，无癥瘕痞块，爪甲色泽红润，双下肢无浮肿，舌质红，苔薄黄略腻，脉滑。"
                      "回答："
                      "【证候】湿热蕴结证\n【病因】是指湿热互结，热不得越，湿不得泄，以身热不扬，口渴不欲多饮，大便泄泻，小便短黄，舌红苔黄腻，脉滑数等为常见症的证候。\n【诊断】狐臭\n【治宜】清热利湿、清热化湿\n【推荐】常见中药有积雪草、贯叶金丝桃、布渣叶等；方用和肝利胆糖浆、强肝糖浆胶囊、结石通胶囊等。"
                      "以上是示例。请学习以上回答，对以下问题进行回答。")
    input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.MC.json"
    # 检查并创建文件夹
    if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
        os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
    output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/MC.{current_time}.result.csv"
# elif question_type == "MC":
#     demo_file_path = ("")
#     input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.MC.json"
#     # 检查并创建文件夹
#     if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
#         os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
#     output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/MC.{current_time}.result.csv"
elif question_type == "instructions":
    demo_file_path = ("问题："
                      "请提供以下中成药的使用说明书：柏子养心丸"
                      "回答："
                      "以下是柏子养心丸的详细说明书：【主要成分】柏子仁、党参、炙黄芪、川芎、当归、茯苓、远志(制)、酸枣仁、五味子(蒸)、朱砂等13味。【性状】本品为棕色的水蜜丸、棕色至棕褐色的小蜜丸或大蜜丸；味先甜而后苦、微麻。【规格】9g*10丸。【功能主治】补气，养血，安神。用于心气虚寒，心悸易惊，失眠多梦，健忘。【用法用量】口服，大蜜丸一次1丸，一日2次。【不良反应】尚不明确。【生产企业】北京御生堂集团石家庄制药有限公司。【注意事项】尚不明确。请仔细阅读说明书并遵医嘱使用。"
                      "以上是示例。请学习以上回答，对以下问题进行回答。")
    input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.instructions.json"
    # 检查并创建文件夹
    if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
        os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
    output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/instructions.{current_time}.result.csv"
# elif question_type == "instructions":
#     demo_file_path = ("")
#     input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.instructions.json"
#     # 检查并创建文件夹
#     if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
#         os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
#     output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/instructions.{current_time}.result.csv"
# elif question_type == "chemical":
#     demo_file_path = ("问题："
#                       "桑枝的化学成分包含哪些？"
#                       "回答："
#                       "桑枝的化学成分主要包括多种黄酮类化合物，例如桑酮、桑素、桑色素、桑色烯素、环桑素及环桑色烯素等。此外，桑枝中还含有槲皮素和山柰酚等成分。同时，桑枝亦含有生物碱、多糖以及香豆素等化合物。这些成分共同赋予了桑枝独特的药用价值和生理活性。"
#                       "以上是示例。请学习以上回答，对以下问题进行回答。")
#     input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.chemical.json"
#     # 检查并创建文件夹
#     if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
#         os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
#     output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/chemical.{current_time}.result.csv"
elif question_type == "chemical":
    demo_file_path = ("问题：桑枝的化学成分包含哪些？回答：桑枝的化学成分主要包括多种黄酮类化合物，例如桑酮、桑素、桑色素、桑色烯素、环桑素及环桑色烯素等。此外，桑枝中还含有槲皮素和山柰酚等成分。同时，桑枝亦含有生物碱、多糖以及香豆素等化合物。这些成分共同赋予了桑枝独特的药用价值和生理活性。"
                      "问题：紫珠叶的化学成分包含哪些？回答：紫珠叶是一种具有多种化学成分的植物，其主要成分包括黄酮类、苯乙醇苷类和三萜类等。其中，黄酮类成分包括紫珠萜酮、木犀草素和芹菜素等；苯乙醇苷类成分以毛蕊花糖苷为主；而三萜类成分则以熊果酸为代表。此外，紫珠叶中还含有甾醇等物质。根据《中国药典》的规定，紫珠叶中毛蕊花糖苷的含量不得低于0.50%。这些化学成分使紫珠叶在传统药用和现代药理研究中具备重要的价值。"
                      "问题：紫花地丁的化学成分包含哪些？回答：紫花地丁的化学成分十分丰富，主要包含黄酮及其苷类、香豆素及其苷类、甾醇、生物碱、内酯和挥发油等。除此之外，它还富含多种微量元素，如钙、钠、钾、锰等，以及一些有机酸。这些成分共同赋予了紫花地丁独特的药用价值和保健功效。"
                      "以上是示例。请学习以上回答，对以下问题进行回答。")
    input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.chemical.json"
    # 检查并创建文件夹
    if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
        os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
    output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/chemical.{current_time}.result.csv"
# elif question_type == "chemical":
#     demo_file_path = ("")
#     input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.chemical.json"
#     # 检查并创建文件夹
#     if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
#         os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
#     output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/chemical.{current_time}.result.csv"
elif question_type == "acupoint":
    demo_file_path = ("问题："
                      "患者主诉头部疼痛，疼痛主要集中在侧头部，且多为单侧出现。该症状可能伴随其他不适，疼痛程度时轻时重，给日常生活带来一定困扰。 请给出治疗的穴位？"
                      "回答："
                      "风池、太阳、率谷、阿是穴、外关、足临泣、风门、列缺、大椎、曲池、偏历、阴陵泉、太冲、侠溪、三阴交、肾俞、太溪、三阴交、气海、足三里、中脘、丰隆、血海、膈俞。"
                      "以上是示例。请学习以上回答，对以下问题进行回答。")
    input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.acupoint.json"
    # 检查并创建文件夹
    if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
        os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
    output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/acupoint.{current_time}.result.csv"
# elif question_type == "acupoint":
#     demo_file_path = ("")
#     input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.acupoint.json"
#     # 检查并创建文件夹
#     if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
#         os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
#     output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/acupoint.{current_time}.result.csv"
elif question_type == "abstract":
    demo_file_path = ("问题："
                      "这是一篇文献的标题，请根据标题写一个摘要。标题：露蕊乌头化学成分及生物活性研究进展"
                      "回答："
                      "露蕊乌头为毛茛科乌头属露蕊乌头亚属唯一植物，常用藏药之一。植物体内含有生物碱、挥发油、酚酸、萜类等化学成分，其中二萜生物碱是主要活性成分，具有镇痛抗炎、抑制肿瘤、强心、免疫调节、抑菌杀虫等作用，可用于治疗风湿、关节疼痛等症。露蕊乌头与常用药用植物乌头、北乌头同为乌头属植物，然而一直以来对其研究较少。为了掌握露蕊乌头的研究现状，笔者根据国内外文献报道，总结并综述了1980—2022年露蕊乌头化学成分及生物活性的研究进展。"
                      "以上是中文示例。请学习以上回答，对以下问题进行回答，如果是中文的标题，请回答中文的摘要。如果是英文的标题，请回答英文的摘要。")
    input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.abstract.json"
    # 检查并创建文件夹
    if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
        os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
    output_file_path = f"/data/1JY/5TCMChat/results2/abstract/{model_type}/abstract.{current_time}.result.csv"
# elif question_type == "abstract":
#     demo_file_path = ("")
#     input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.abstract.json"
#     # 检查并创建文件夹
#     if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
#         os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
#     output_file_path = f"/data/1JY/5TCMChat/results2/abstract/{model_type}/abstract.{current_time}.result.csv"
elif question_type == "title":
    demo_file_path = ("问题："
                      "这是一篇文献的摘要，请根据摘要写一个标题。摘要：目的：观察益气活血方及其拆方治疗冠心病心力衰竭气虚血瘀证的临床疗效。方法：将160例冠心病心力衰竭气虚血瘀证患者随机分成对照组、益气组、活血组、益气活血组，每组40例。结论：在西药规范化治疗基础上运用益气活血方及其拆方，可提高气虚血瘀型冠心病心力衰竭患者的临床疗效，并具有良好的安全性，其中以益气活血方效果最佳。 "
                      "回答："
                      "益气活血方及其拆方治疗冠心病心力衰竭气虚血瘀证的双盲随机对照试验"
                      "以上是中文示例。请学习以上回答，对以下问题进行回答，如果是中文的摘要，请回答中文的标题。如果是英文的摘要，请回答英文的标题。")
    input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.title.json"
    # 检查并创建文件夹
    if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
        os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
    output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/title.{current_time}.result.csv"
# elif question_type == "title":
#     demo_file_path = ("")
#     input_file_path = "/data/1JY/5TCMChat/evaluation/all/QA.test.title.json"
#     # 检查并创建文件夹
#     if not os.path.exists(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/"):
#         os.makedirs(f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/")
#     output_file_path = f"/data/1JY/5TCMChat/results2/{question_type}/{model_type}/title.{current_time}.result.csv"
else:
    raise ValueError(f"文件路径出问题。Unsupported question type: {question_type}")

#构建模型名字或路径
if model_type == "DeepSeek-R1-Distill-Qwen-7B":
    model_namepath = "/data/1JY/98model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
elif model_type == "DeepSeek-R1-Distill-Qwen-14B":
    model_namepath = "/data/1JY/98model/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
elif model_type == "DeepSeek-R1-Distill-Qwen-32B":
    model_namepath = "/data/1JY/98model/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
elif model_type == "Qwen2.5-Math-7B":
    model_namepath = "/data/1JY/98model/Qwen/Qwen2___5-Math-7B"
elif model_type == "Qwen2.5-14B-Instruct":
    model_namepath = "/data/1JY/98model/Qwen/Qwen2___5-14B-Instruct"
elif model_type == "Qwen2.5-32B-Instruct":
    model_namepath = "/data/1JY/98model/Qwen/Qwen2___5-32B-Instruct"
elif model_type == "Qwen2.5-72B-Instruct":
    model_namepath = "/data/1JY/98model/Qwen/Qwen2___5-72B-Instruct"
elif model_type == "Baichuan-M1-14B-Instruct":
    model_namepath = "/data/1JY/98model/baichuan-inc/Baichuan-M1-14B-Instruct"
elif model_type == "Baichuan2-7B-Chat":
    model_namepath = "/data/1JY/98model/baichuan-inc/Baichuan2-7B-Chat"
elif model_type == "Baichuan2-13B-Chat":
    model_namepath = "/data/1JY/98model/baichuan-inc/Baichuan2-13B-Chat"
elif model_type == "chatglm3-6b":
    model_namepath = "/data/1JY/98model/ZhipuAI/chatglm3-6b"
elif model_type == "Llama3-8B-Chinese-Chat":
    model_namepath = "/data/1JY/98model/LLM-Research/Llama3-8B-Chinese-Chat"
elif model_type == "BianQue-2":
    model_namepath = "/data/1JY/98model/AI-ModelScope/BianQue-2"
elif model_type == "chatmed":
    model_namepath = "/data/1JY/98model/huggyllama/huggyllamallama-7b"
    lora_path = "/data/1JY/98model/michaelwzhu/ChatMed-Consult"
elif model_type == "Lingdan-13B-Base":
    model_namepath = "/data/1JY/98model/TCMLLM/Lingdan-13B-Base"
elif model_type == "Lingdan-13B-PR":
    model_namepath = "/data/1JY/98model/TCMLLM/Lingdan-13B-PR"
elif model_type == "HuatuoGPT2-13B":
    model_namepath = "/data/1JY/98model/FreedomIntelligence/HuatuoGPT2-13B"
elif model_type == "Sunsimiao-7B":
    model_namepath = "/data/1JY/98model/X-D-Lab/Sunsimiao-Qwen2-7B"
elif model_type == "BianCang-Qwen2.5-7B-Instruct":
    model_namepath = "/data/1JY/98model/QLUNLP/BianCang-Qwen2___5-7B-Instruct"
elif model_type == "TCMchat":
    model_namepath = "/data/1JY/98model/joshuaHe/tcm-chat"
elif model_type == "ZhongjingGPT1_13B":
    model_namepath = "/data/1JY/98model/baichuan-inc/Baichuan2-13B-Chat"
    lora_path = "/data/1JY/98model/CMLM/ZhongjingGPT1_13B"
elif model_type == "Bentao":
    model_namepath = "/data/1JY/98model/huggyllama/huggyllamallama-7b"
    lora_path = "/data/1JY/98model/thinksoso/lora-llama-med"
elif model_type == "ShenNong":
    model_namepath = "/data/1JY/98model/huggyllama/huggyllamallama-7b"
    lora_path = "/data/1JY/98model/michaelwzhu/ShenNong-TCM-LLM"
elif model_type == "gpt-3.5-turbo":
    model_namepath = "gpt-3.5-turbo"
    base_url = "https://chatapi.littlewheat.com/v1"
    api_key = "sk-zYEOwXkxuXIysPu0q1pPwRA4TVlCLm0Jnxo8oC5i5rjw1HIi"
elif model_type == "gpt-4o":
    model_namepath = "gpt-4o"
    base_url = "https://chatapi.littlewheat.com/v1"
    api_key = "sk-O6ramjcste3LqoyjhqKQkoaYVKVLjXP3yfWJ3ZTWVl7y74Ev"
elif model_type == "o1":
    model_namepath = "o1"
    base_url = "https://api.aigc369.com/v1"
    api_key = "sk-4KJoBIO7ZeEHJ401fFZXwkPq3zkQgZeLgly9n6bZOzJ07Ed1"
elif model_type == "deepseek-r1":
    model_namepath = "deepseek-r1"
    base_url = "https://chatapi.littlewheat.com/v1"
    api_key = "sk-aNIeJY8zY8K5Xb1izo0c3gEWenxRyp2eOhUzH7rXdHa7zAUp"
elif model_type == "deepseek-v3":
    model_namepath = "deepseek-v3"
    base_url = "https://chatapi.littlewheat.com/v1"
    api_key = "sk-aNIeJY8zY8K5Xb1izo0c3gEWenxRyp2eOhUzH7rXdHa7zAUp"

elif model_type == "our_model-7B":
    model_namepath = "/data/1JY/6.DeepseekFT/saves/Qwen2.5-Math-7B/merge.SFT/train.128.256.0.2.2"
elif model_type == "our_model-14B":
    model_namepath = "/data/1JY/98model/baichuan-inc/Baichuan2-13B-Chat"
elif model_type == "our_model-32B":
    model_namepath = "/data/1JY/98model/ZhipuAI/chatglm3-6b"

elif model_type == "14B.128.256.0.2.4":
    model_namepath = "/data/1JY/6.DeepseekFT/saves/Qwen2.5-Math-14B/merge.SFT/train.128.256.0.2.4"
elif model_type == "14B.64.128.0.2.4":
    model_namepath = "/data/1JY/6.DeepseekFT/saves/Qwen2.5-Math-14B/merge.SFT/train.64.128.0.2.4"
elif model_type == "14B.32.64.0.2.4":
    model_namepath = "/data/1JY/6.DeepseekFT/saves/Qwen2.5-Math-14B/merge.SFT/train.32.64.0.2.4"
elif model_type == "14B.16.32.0.2.4":
    model_namepath = "/data/1JY/6.DeepseekFT/saves/Qwen2.5-Math-14B/merge.SFT/train.16.32.0.2.4"
elif model_type == "14B.8.16.0.2.4":
    model_namepath = "/data/1JY/6.DeepseekFT/saves/Qwen2.5-Math-14B/merge.SFT/train.8.16.0.2.4"

elif model_type == "14B.128.256.0.0.4":
    model_namepath = "/data/1JY/6.DeepseekFT/saves/Qwen2.5-Math-14B/merge.SFT/train.128.256.0.0.4"
elif model_type == "14B.128.256.0.2.2":
    model_namepath = "/data/1JY/6.DeepseekFT/saves/Qwen2.5-Math-14B/merge.SFT/train.128.256.0.2.2"
elif model_type == "14B.128.256.0.2.4.256":
    model_namepath = "/data/1JY/6.DeepseekFT/saves/Qwen2.5-Math-14B/merge.SFT/train.128.256.0.2.4.256"
elif model_type == "14B.128.256.0.2.4.512":
    model_namepath = "/data/1JY/6.DeepseekFT/saves/Qwen2.5-Math-14B/merge.SFT/train.128.256.0.2.4.512"
elif model_type == "14B.128.256.0.2.4.1024":
    model_namepath = "/data/1JY/6.DeepseekFT/saves/Qwen2.5-Math-14B/merge.SFT/train.128.256.0.2.4.1024"
elif model_type == "14B.128.256.0.2.6":
    model_namepath = "/data/1JY/6.DeepseekFT/saves/Qwen2.5-Math-14B/merge.SFT/train.128.256.0.2.6"
elif model_type == "14B.128.256.0.4.4":
    model_namepath = "/data/1JY/6.DeepseekFT/saves/Qwen2.5-Math-14B/merge.SFT/train.128.256.0.4.4"

else:
    raise ValueError(f"模型名字或路径出问题。Unsupported model_type: {model_type}")

# 构建system_prompt
def create_system_prompt(file_path, question_type):
    """
    从CSV文件中读取数据并生成system_prompt。

    参数:
    file_path (str): CSV文件的路径。
    question_type (str): 回答类型，"choice" 或 "discuss"。

    返回:
    str: 生成的system_prompt。
    """
    if question_type == "choice":
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 合并列
        df['merged'] = df['question'] + '\n' + '选项：' + '\n' + 'A：' + df['A'] + '\n' + 'B：' + df['B'] + '\n' + 'C：' + df[
            'C'] + '\n' + 'D：' + df['D'] + '\n' + 'E：' + df['E'] + '\n' + '正确选项：' + df['answer']

        # 获取第一行的合并内容
        few_shot1 = df['merged'][0]

        # 构建system_prompt
        system_prompt = f'"{few_shot1}"\n以上是示例。请学习以上回答，只有一个选项是正确选项，只输出正确选项，比如A、B、C、D、E，不要输出解释和思考。\n'
    else:
        system_prompt = file_path

    return system_prompt

# 构建user_input
def create_user_input(file_path,question_type):
    """
    从CSV文件中读取数据并生成user_input。

    参数:
    file_path (str): CSV文件的路径。

    返回:
    str: 生成的user_input。
    """
    if question_type == "choice":
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 合并列
        df['merged'] = df['question'] + '\n' + '选项：' + '\n' + 'A：' + df['A'] + '\n' + 'B：' + df['B'] + '\n' + 'C：' + df[
            'C'] + '\n' + 'D：' + df['D'] + '\n' + 'E：' + df['E']

        # 获取第一行的合并内容
        user_input = df['merged']

    elif question_type in ["QA","RC","NER","MC","recommend","instructions",
                           "chemical","pharmacological","abstract","title","acupoint"]:

        # 打开并读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            df = json.load(file)

        # 删除前298个元素
        #df = df[939:]

        # 合并 "instruction" 和 "input" 为 user_input 并保留该字段
        user_input = []
        for item in df:
            if 'instruction' in item and 'input' in item:
                user = item['instruction'] + item['input']
                user_input.append(user)
        df = pd.DataFrame(df)
    else:
        raise ValueError(f"构建user_input出问题。Unsupported question type: {question_type}")
    return user_input, df

# 逐行生成回答
def process_responses_qwen(model, tokenizer, system_prompt, user_input, rawdata, output_file_path):
    # 生成回答并合并数据
    for i, input_text in enumerate(user_input):
        response = response_Qwen(model, tokenizer, system_prompt, input_text)
        print(f"问题 {i + 1}: {input_text}\n回答: {response}\n")

        # 将 response 添加到 rawdata 中
        rawdata.at[i, 'candidate'] = response

        # 保存结果
        rawdata.to_csv(output_file_path, index=False)
        #rawdata.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f"结果已保存到 {output_file_path}")

def process_responses_baichuan2(model, tokenizer, system_prompt, user_input, rawdata, output_file_path):
    # 生成回答并合并数据
    for i, input_text in enumerate(user_input):
        response = response_baichuan2(model, tokenizer, system_prompt, input_text)
        print(f"问题 {i + 1}: {input_text}\n回答: {response}\n")

        # 将 response 添加到 rawdata 中
        rawdata.at[i, 'candidate'] = response

        # 保存结果
        rawdata.to_csv(output_file_path, index=False)
        print(f"结果已保存到 {output_file_path}")

def process_responses_chatglm3(model, tokenizer, system_prompt, user_input, rawdata, output_file_path):
    # 生成回答并合并数据
    for i, input_text in enumerate(user_input):
        response = response_chatglm3(model, tokenizer, system_prompt, input_text)
        print(f"问题 {i + 1}: {input_text}\n回答: {response}\n")

        # 将 response 添加到 rawdata 中
        rawdata.at[i, 'candidate'] = response

        # 保存结果
        rawdata.to_csv(output_file_path, index=False)
        print(f"结果已保存到 {output_file_path}")

def process_responses_bianque2(pipe, system_prompt, user_input, rawdata, output_file_path):
    # 生成回答并合并数据
    for i, input_text in enumerate(user_input):
        response = response_bianque2(pipe, system_prompt, input_text)
        print(f"问题 {i + 1}: {input_text}\n回答: {response}\n")

        # 将 response 添加到 rawdata 中
        rawdata.at[i, 'candidate'] = response

        # 保存结果
        rawdata.to_csv(output_file_path, index=False)
        print(f"结果已保存到 {output_file_path}")

def process_responses_Llama3(model, tokenizer, system_prompt, user_input, rawdata, output_file_path):
    # 生成回答并合并数据
    for i, input_text in enumerate(user_input):
        response = response_Llama3(model, tokenizer, system_prompt, input_text)
        print(f"问题 {i + 1}: {input_text}\n回答: {response}\n")

        # 将 response 添加到 rawdata 中
        rawdata.at[i, 'candidate'] = response

        # 保存结果
        rawdata.to_csv(output_file_path, index=False)
        print(f"结果已保存到 {output_file_path}")

def process_responses_LLaMA7B(model, tokenizer, system_prompt, user_input, rawdata, output_file_path):
    # 生成回答并合并数据
    for i, input_text in enumerate(user_input):
        response = response_LLaMA7B(model, tokenizer, system_prompt, input_text)
        print(f"回答{i + 1}: {response}\n")

        # 将 response 添加到 rawdata 中
        rawdata.at[i, 'candidate'] = response

        # 保存结果
        rawdata.to_csv(output_file_path, index=False)
        print(f"结果已保存到 {output_file_path}")

def process_responses_Lingdan(model, tokenizer, system_prompt, user_input, rawdata, output_file_path):
    # 生成回答并合并数据
    for i, input_text in enumerate(user_input):
        response = response_Lingdan(model, tokenizer, system_prompt, input_text)
        print(f"问题 {i + 1}: {input_text}\n回答: {response}\n")

        # 将 response 添加到 rawdata 中
        rawdata.at[i, 'candidate'] = response

        # 保存结果
        rawdata.to_csv(output_file_path, index=False)
        print(f"结果已保存到 {output_file_path}")

def process_responses_api(base_url, api_key, model_namepath, system_prompt, user_input, rawdata, output_file_path):
    # 生成回答并合并数据
    for i, input_text in enumerate(user_input):
        response = response_API(base_url, api_key, model_namepath, system_prompt, input_text)
        print(f"问题 {i + 1}: {input_text}\n回答: {response}\n")

        # 将 response 添加到 rawdata 中
        rawdata.at[i, 'candidate'] = response

        # 保存结果
        rawdata.to_csv(output_file_path, index=False)
        print(f"结果已保存到 {output_file_path}")

# 主函数
def main():
    # system_prompt = create_system_prompt(demo_file_path, question_type)
    system_prompt = demo_file_path
    print("系统提示词：", system_prompt)
    user_input, rawdata = create_user_input(input_file_path, question_type)

    # 构建模型名字或路径
    if model_type in ["DeepSeek-R1-Distill-Qwen-7B","DeepSeek-R1-Distill-Qwen-14B","DeepSeek-R1-Distill-Qwen-32B",
                      "Qwen2.5-Math-7B","Qwen2.5-14B-Instruct","Qwen2.5-32B-Instruct","Qwen2.5-72B-Instruct","Baichuan-M1-14B-Instruct",
                      "Sunsimiao-7B","BianCang-Qwen2.5-7B-Instruct","our_model-7B","our_model-14B","our_model-32B",
                      "14B.128.256.0.2.4","14B.64.128.0.2.4","14B.32.64.0.2.4","14B.16.32.0.2.4","14B.8.16.0.2.4",
                      "14B.128.256.0.0.4","14B.128.256.0.2.2","14B.128.256.0.2.6","14B.128.256.0.4.4",
                      "14B.128.256.0.2.4.256","14B.128.256.0.2.4.512","14B.128.256.0.2.4.1024"]:
        # 加载模型和分词器
        model, tokenizer = model_tokenizer_Qwen(model_namepath)
        # 逐行生成回答
        process_responses_qwen(model, tokenizer, system_prompt, user_input, rawdata, output_file_path)

    elif model_type in ["Baichuan2-7B-Chat","Baichuan2-13B-Chat","HuatuoGPT2-13B","TCMchat"] :
        # 加载模型和分词器
        model, tokenizer = model_tokenizer_baichuan2(model_namepath)
        # 逐行生成回答
        process_responses_baichuan2(model, tokenizer, system_prompt, user_input, rawdata, output_file_path)

    elif model_type in ["ZhongjingGPT1_13B"] :
        # 加载模型和分词器
        model, tokenizer = model_tokenizer_lora_baichuan2(model_namepath,lora_path)
        # 生成回答并合并数据
        process_responses_baichuan2(model, tokenizer, system_prompt, user_input, rawdata, output_file_path)

    elif model_type in ["chatglm3-6b"] :
        # 加载模型和分词器
        model, tokenizer = model_tokenizer_chatglm3(model_namepath)
        # 逐行生成回答
        process_responses_chatglm3(model, tokenizer, system_prompt, user_input, rawdata, output_file_path)

    elif model_type in ["Llama3-8B-Chinese-Chat"] :
        # 加载模型和分词器
        model, tokenizer = model_tokenizer_Llama3(model_namepath)
        # 逐行生成回答
        process_responses_Llama3(model, tokenizer, system_prompt, user_input, rawdata, output_file_path)

    elif model_type in ["BianQue-2"] :
        # 加载模型和分词器
        pipe = model_tokenizer_bianque2(model_namepath)
        # 逐行生成回答
        process_responses_bianque2(pipe, system_prompt, user_input, rawdata, output_file_path)

    elif model_type in ["chatmed","Bentao","ShenNong"] :
        # 加载模型和分词器
        model, tokenizer = model_tokenizer_lora_LLaMA7B(model_namepath,lora_path)
        # 生成回答并合并数据
        process_responses_LLaMA7B(model, tokenizer, system_prompt, user_input, rawdata, output_file_path)

    elif model_type in ["Lingdan-13B-Base","Lingdan-13B-PR"] :
        # 加载模型和分词器
        model, tokenizer = model_tokenizer_Lingdan(model_namepath)
        # 逐行生成回答
        process_responses_Lingdan(model, tokenizer, system_prompt, user_input, rawdata, output_file_path)

    elif model_type in ["gpt-3.5-turbo","gpt-4o","o1","deepseek-r1","deepseek-v3"] :
        # 逐行生成回答
        process_responses_api(base_url, api_key, model_namepath, system_prompt, user_input, rawdata,
                                 output_file_path)

    else:
        raise ValueError(f"main出问题。Unsupported model_type: {model_type}")

# 调用主函数
if __name__ == "__main__":
    main()


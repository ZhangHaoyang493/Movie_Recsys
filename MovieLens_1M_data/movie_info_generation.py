from openai import OpenAI
import json
from tqdm import tqdm
import os

# 替换为你自己的 API Key（从 ECNU 获取）
API_KEY = os.getenv("ECNU_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://chat.ecnu.edu.cn/open/api/v1"  # 注意：不包含 /chat/completions
)


def get_movie_details(movie_name):
    prompt = f"""
                你是一位专业的电影数据查询助手。请根据用户提供的电影英文名称 `{movie_name}`，查询并返回该电影的详细信息。

                **你需要返回的数据字段包括**：
                - `English_title`: 电影英文原名
                - `Chinese_title`: 电影中文译名
                - `Chinese_description`: 电影中文剧情简介（400字左右）
                - `tags`: 电影类型标签列表（中文，如 ["剧情", "科幻", "动作"]）
                - `director_chinese_name`: 导演中文姓名
                - `direction_English_name`: 导演英文姓名
                - `cast_English_name`: 主演英文姓名列表（最多5位主要演员）
                - `cast_Chinese_name`: 主演中文姓名列表（最多5位主要演员）
                - `country`: 制片国家/地区（中文）
                - `language`: 电影语言（中文）
                - `release_year`: 上映年份（YYYY格式）
                - `duration`: 片长（分钟数）

                **输出格式要求**：
                - 严格按照 JSON 格式输出
                - 只输出纯 JSON 数据，不包含任何解释、说明或代码块标记
                - 如果某个字段信息无法获取，请使用 `null` 或空数组 `[]` 表示
                - 确保 JSON 结构完整，格式正确

                **输出示例**：
                {{
                "English_title": "电影英文名称",
                "Chinese_title": "电影中文名称",
                "Chinese_description": "电影中文简介",
                "tags": ["标签1", "标签2"],
                "director_chinese_name": "导演中文名称",
                "direction_English_name": "导演英文名称",
                "cast_English_name": ["主演英文名称1", "主演英文名称2"],
                "cast_Chinese_name": ["主演中文名称1", "主演中文名称2"],
                "country": "制片国家/地区",
                "language": "语言",
                "release_year": "上映时间",
                "duration": "片长"
                }}
"""

    response = client.chat.completions.create(
        model="ecnu-plus",
        messages=[
            {"role": "system", "content": "你是一个电影信息专家。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=5000

    )
    json_str = response.choices[0].message.content
    data = json.loads(json_str)

    return data

with open("movie_details.json", "r", encoding="utf-8") as f:
    movie_details = json.load(f)

with open("/data2/zhy/Movie_Recsys/MovieLens_1M_data/movies.dat", "r", encoding='ISO-8859-1') as f:
    lines = f.readlines()
    idx = movie_details['idx']  # idx代表该处理第几条数据了

    for i, line in tqdm(enumerate(lines)):
        if i < movie_details['idx']:
            continue
        movie_id, movie_title, movie_genres = line.strip().split("::")
        try:
            movie_data = get_movie_details(movie_title)
        except:
            try:
                movie_data = get_movie_details(movie_title)
            except:
                movie_data = {
                    "English_title": movie_title,
                    "Chinese_title": None,
                    "Chinese_description": None,
                    "tags": [],
                    "director_chinese_name": None,
                    "direction_English_name": None,
                    "cast_English_name": [],
                    "cast_Chinese_name": [],
                    "country": None,
                    "language": None,
                    "release_year": None,
                    "duration": None
                }
        movie_details[movie_id] = movie_data
        movie_details['idx'] = movie_details['idx'] + 1

        with open("movie_details.json", "w", encoding="utf-8") as f:
            json.dump(movie_details, f, ensure_ascii=False, indent=4)
        idx += 1



import pandas as pd
import requests
import time
import os

TENCENT_KEY = "SA4BZ-MYB67-DV5XS-P4OFW-GLUMK-ZUBHK"


def reverse_geocode_tencent(lon, lat):
    """腾讯地图逆地理编码"""
    url = "https://apis.map.qq.com/ws/geocoder/v1/"
    poi_opts = ";".join([
        "address_format=short",
        "radius=1000",
        "page_size=15",
        "policy=4",  # 社交签到场景，优先热门地点
        "category=购物场所,餐饮服务,地铁站,医院,银行,学校"
    ])
    params = {
        "key": TENCENT_KEY,
        "location": f"{lat},{lon}",  # 腾讯地图格式：lat,lon
        "get_poi": 1,
        "poi_options": poi_opts,
        "output": "json"
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") != 0:
            print("反查失败:", data.get("message"), lon, lat)
            return None
        return data["result"]
    except Exception as e:
        print("请求异常:", e, lon, lat)
        return None


def summarize_pois(result):
    """汇总POI信息"""
    if not result:
        return "", "", ""

    # 优先来源：famous_area, landmark_l1, landmark_l2
    fa = result.get("famous_area", {}).get("title", "")
    lm1 = result.get("landmark_l1", {}).get("title", "")
    lm2 = result.get("landmark_l2", {}).get("title", "")

    # 聚合POI类别数量
    cat_count = {}
    for poi in result.get("pois", []):
        cat = poi.get("category", "").split(":")[0]
        if cat:
            cat_count.setdefault(cat, []).append(poi.get("title", ""))

    # 排序选最频繁两个大类
    top_cats = sorted(cat_count.items(), key=lambda kv: len(kv[1]), reverse=True)[:2]
    top_names = []
    for cat, names in top_cats:
        if names:
            top_names.append(names[0])

    # 综合summary: 首选fa/lm1/lm2，再加代表性POI
    summary_parts = [x for x in [fa, lm1, lm2] if x]
    summary_parts += top_names
    summary = ";".join(summary_parts)

    return fa, ";".join(top_names), summary


def main():
    # 修改文件路径格式
    data_dir = '../data'
    input_file = os.path.join(data_dir, 'grid_coordinates.csv')
    output_file = os.path.join(data_dir, 'grid_coordinates_with_summary.csv')

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到输入文件 {input_file}")
        print("请确认文件路径是否正确")
        return

    try:
        df = pd.read_csv(input_file)
        print(f"成功读取文件，共 {len(df)} 行数据")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 计算网格中心点坐标
    if 'grid_lat_min' in df.columns and 'grid_lon_min' in df.columns:
        df["latitude"] = (df["grid_lat_min"] + df["grid_lat_max"]) / 2
        df["longitude"] = (df["grid_lon_min"] + df["grid_lon_max"]) / 2
        print("已计算网格中心点坐标")
    elif 'latitude' not in df.columns or 'longitude' not in df.columns:
        print("错误：缺少必要的坐标列")
        return

    # 新列初始化
    new_columns = ["formatted_address", "province", "city", "district",
                   "township", "street", "famous_area", "top_poi", "pois_summary"]
    for col in new_columns:
        df[col] = ""

    total_rows = len(df)
    success_count = 0

    for idx, row in df.iterrows():
        try:
            lon, lat = row["longitude"], row["latitude"]
            print(f"处理第 {idx + 1}/{total_rows} 个网格: ({lat:.6f}, {lon:.6f})")

            result = reverse_geocode_tencent(lon, lat)
            if not result:
                continue

            # 地址信息
            addr_cmp = result.get("address_component", {})
            df.at[idx, "formatted_address"] = result.get("address", "")
            df.at[idx, "province"] = addr_cmp.get("province", "")
            df.at[idx, "city"] = addr_cmp.get("city", "")
            df.at[idx, "district"] = addr_cmp.get("district", "")
            df.at[idx, "township"] = addr_cmp.get("street", "")
            df.at[idx, "street"] = addr_cmp.get("street_number", "")

            # POI信息汇总
            fa, top, summary = summarize_pois(result)
            df.at[idx, "famous_area"] = fa
            df.at[idx, "top_poi"] = top
            df.at[idx, "pois_summary"] = summary

            success_count += 1

            if (idx + 1) % 10 == 0:
                print(f"已处理 {idx + 1}/{total_rows} 网格，成功 {success_count} 个")

            time.sleep(0.2)  # API调用间隔

        except Exception as e:
            print(f"处理第 {idx + 1} 行数据时出错: {e}")
            continue

    # 保存结果
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"处理完成：保存至 {output_file}")
        print(f"总计处理 {total_rows} 个网格，成功 {success_count} 个")
    except Exception as e:
        print(f"保存文件失败: {e}")


if __name__ == "__main__":
    main()

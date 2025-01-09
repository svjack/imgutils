# imgutils

[![PyPI](https://img.shields.io/pypi/v/dghs-imgutils)](https://pypi.org/project/dghs-imgutils/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dghs-imgutils)
![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/narugo1992/8bfaa96eaa25cc9dac54debbf22d363d/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/narugo1992/8bfaa96eaa25cc9dac54debbf22d363d/raw/comments.json)

[![Code Test](https://github.com/deepghs/imgutils/workflows/Code%20Test/badge.svg)](https://github.com/deepghs/imgutils/actions?query=workflow%3A%22Code+Test%22)
[![Package Release](https://github.com/deepghs/imgutils/workflows/Package%20Release/badge.svg)](https://github.com/deepghs/imgutils/actions?query=workflow%3A%22Package+Release%22)
[![codecov](https://codecov.io/gh/deepghs/imgutils/branch/main/graph/badge.svg?token=XJVDP4EFAT)](https://codecov.io/gh/deepghs/imgutils)

[![Discord](https://img.shields.io/discord/1157587327879745558?style=social&logo=discord&link=https%3A%2F%2Fdiscord.gg%2FTwdHJ42N72)](https://discord.gg/TwdHJ42N72)
![GitHub Org's stars](https://img.shields.io/github/stars/deepghs)
[![GitHub stars](https://img.shields.io/github/stars/deepghs/imgutils)](https://github.com/deepghs/imgutils/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/deepghs/imgutils)](https://github.com/deepghs/imgutils/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/deepghs/imgutils)
[![GitHub issues](https://img.shields.io/github/issues/deepghs/imgutils)](https://github.com/deepghs/imgutils/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/deepghs/imgutils)](https://github.com/deepghs/imgutils/pulls)
[![Contributors](https://img.shields.io/github/contributors/deepghs/imgutils)](https://github.com/deepghs/imgutils/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/deepghs/imgutils)](https://github.com/deepghs/imgutils/blob/master/LICENSE)

A convenient and user-friendly anime-style image data processing library that integrates various advanced anime-style
image processing models.

## Installation

You can simply install it with `pip` command line from the official PyPI site.

```shell
pip install dghs-imgutils
```

If your operating environment includes a available GPU, you can use the following installation command to achieve higher
performance:

```shell
pip install dghs-imgutils[gpu]
```

For more information about installation, you can refer
to [Installation](https://deepghs.github.io/imgutils/main/tutorials/installation/index.html).

## Benchmark on Character Recognize Tag on Genshin Imapct some Characters (svjack/Genshin-Impact-Illustration)

```python
from datasets import load_dataset
import os
from PIL import Image

# 加载数据集
Genshin_Impact_Illustration_ds = load_dataset("svjack/Genshin-Impact-Illustration")["train"]
ds_size = len(Genshin_Impact_Illustration_ds)
name_image_dict = {}
for i in range(ds_size):
    row_dict = Genshin_Impact_Illustration_ds[i]
    name_image_dict[row_dict["name"]] = row_dict["image"]

# 保存图片的函数
def save_images(name_image_dict, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历字典，保存图片
    for name, image in name_image_dict.items():
        # 构造文件路径
        file_path = os.path.join(output_dir, f"{name}.png")  # 假设保存为 PNG 格式
        # 保存图片
        image.save(file_path)
        print(f"Saved {file_path}")

# 示例：保存图片到指定路径
output_directory = "genshin_impact_images"  # 替换为你想保存图片的路径
save_images(name_image_dict, output_directory)
```

```bash
python score_tag_script.py genshin_impact_images genshin_impact_images_with_tags
```

```python
import pathlib
import pandas as pd
import numpy as np
def rj_func(x):
    import json 
    with open(x, "r") as f:
        return json.load(f)

df = pd.DataFrame(
pd.Series(
    list(pathlib.Path("genshin_impact_images_with_tags/").rglob("*.json"))
).map(str).map(lambda x: np.nan if ".ipy" in x else x).dropna().map(
    lambda x: (x.split("/")[-1].replace(".json", ""), rj_func(x)["characters"])
).values.tolist())
df.columns = ["name", "person"]
df["detected_name"] = df["person"].map(lambda x: list(x.items())).map(lambda x: sorted(x, key = lambda t2: -1 * t2[1]))
df["detected_name"] = df["detected_name"].map(lambda x: x[0] if x else np.nan)
df_np = df.iloc[np.where(pd.isna(df))[0]].copy()
df = df.dropna()
df["detected_score"] = df["detected_name"].map(lambda x: x[1])
df["detected_name"] = df["detected_name"].map(lambda x: x[0])
del df["person"]
df = df.sort_values(by = "detected_score", ascending = False)
df = df.reset_index().iloc[:, 1:]
print(df.to_markdown())
```

- Recognized

|    | name       | detected_name                |   detected_score |
|---:|:-----------|:-----------------------------|-----------------:|
|  0 | 迪奥娜     | diona_(genshin_impact)       |         0.996558 |
|  1 | 克洛琳德   | clorinde_(genshin_impact)    |         0.995989 |
|  2 | 诺艾尔     | noelle_(genshin_impact)      |         0.995388 |
|  3 | 久岐忍     | kuki_shinobu                 |         0.994924 |
|  4 | 鹿野院平藏 | shikanoin_heizou             |         0.994901 |
|  5 | 罗莎莉亚   | rosaria_(genshin_impact)     |         0.994056 |
|  6 | 柯莱       | collei_(genshin_impact)      |         0.993521 |
|  7 | 阿蕾奇诺   | arlecchino_(genshin_impact)  |         0.993432 |
|  8 | 九条裟罗   | kujou_sara                   |         0.993423 |
|  9 | 妮露       | nilou_(genshin_impact)       |         0.992874 |
| 10 | 绮良良     | kirara_(genshin_impact)      |         0.992663 |
| 11 | 凝光       | ningguang_(genshin_impact)   |         0.992605 |
| 12 | 魈         | xiao_(genshin_impact)        |         0.992349 |
| 13 | 北斗       | beidou_(genshin_impact)      |         0.992224 |
| 14 | 香菱       | xiangling_(genshin_impact)   |         0.991925 |
| 15 | 早柚       | sayu_(genshin_impact)        |         0.991844 |
| 16 | 凯亚       | kaeya_(genshin_impact)       |         0.991808 |
| 17 | 甘雨       | ganyu_(genshin_impact)       |         0.991792 |
| 18 | 荒泷一斗   | arataki_itto                 |         0.991608 |
| 19 | 枫原万叶   | kaedehara_kazuha             |         0.991601 |
| 20 | 丽莎       | lisa_(genshin_impact)        |         0.991569 |
| 21 | 珊瑚宫心海 | sangonomiya_kokomi           |         0.991537 |
| 22 | 琴         | jean_(genshin_impact)        |         0.99133  |
| 23 | 夜兰       | yelan_(genshin_impact)       |         0.990918 |
| 24 | 那维莱特   | neuvillette_(genshin_impact) |         0.990562 |
| 25 | 雷泽       | razor_(genshin_impact)       |         0.990388 |
| 26 | 可莉       | klee_(genshin_impact)        |         0.990335 |
| 27 | 琳妮特     | lynette_(genshin_impact)     |         0.98981  |
| 28 | 流浪者     | wanderer_(genshin_impact)    |         0.9897   |
| 29 | 卡维       | kaveh_(genshin_impact)       |         0.989697 |
| 30 | 林尼       | lyney_(genshin_impact)       |         0.989454 |
| 31 | 艾尔海森   | alhaitham_(genshin_impact)   |         0.989404 |
| 32 | 莱依拉     | layla_(genshin_impact)       |         0.989404 |
| 33 | 菲谢尔     | fischl_(genshin_impact)      |         0.989271 |
| 34 | 五郎       | gorou_(genshin_impact)       |         0.98924  |
| 35 | 神里绫华   | kamisato_ayaka               |         0.989221 |
| 36 | 芭芭拉     | barbara_(genshin_impact)     |         0.988973 |
| 37 | 胡桃       | hu_tao_(genshin_impact)      |         0.988538 |
| 38 | 雷电将军   | raiden_shogun                |         0.988526 |
| 39 | 七七       | qiqi_(genshin_impact)        |         0.988514 |
| 40 | 温迪       | venti_(genshin_impact)       |         0.988426 |
| 41 | 八重神子   | yae_miko                     |         0.988334 |
| 42 | 纳西妲     | nahida_(genshin_impact)      |         0.988111 |
| 43 | 砂糖       | sucrose_(genshin_impact)     |         0.987847 |
| 44 | 申鹤       | shenhe_(genshin_impact)      |         0.987684 |
| 45 | 行秋       | xingqiu_(genshin_impact)     |         0.987565 |
| 46 | 闲云       | xianyun_(genshin_impact)     |         0.98728  |
| 47 | 云堇       | yun_jin_(genshin_impact)     |         0.986846 |
| 48 | 娜维娅     | navia_(genshin_impact)       |         0.986794 |
| 49 | 莫娜       | mona_(genshin_impact)        |         0.986562 |
| 50 | 托马       | thoma_(genshin_impact)       |         0.986532 |
| 51 | 宵宫       | yoimiya_(genshin_impact)     |         0.986046 |
| 52 | 莱欧斯利   | wriothesley_(genshin_impact) |         0.985957 |
| 53 | 珐露珊     | faruzan_(genshin_impact)     |         0.985146 |
| 54 | 神里绫人   | kamisato_ayato               |         0.985136 |
| 55 | 达达利亚   | tartaglia_(genshin_impact)   |         0.984887 |
| 56 | 迪希雅     | dehya_(genshin_impact)       |         0.984819 |
| 57 | 阿贝多     | albedo_(genshin_impact)      |         0.984713 |
| 58 | 刻晴       | keqing_(genshin_impact)      |         0.984672 |
| 59 | 优菈       | eula_(genshin_impact)        |         0.983777 |
| 60 | 赛诺       | cyno_(genshin_impact)        |         0.982358 |
| 61 | 安柏       | amber_(genshin_impact)       |         0.982014 |
| 62 | 提纳里     | tighnari_(genshin_impact)    |         0.981816 |
| 63 | 迪卢克     | diluc_(genshin_impact)       |         0.981325 |
| 64 | 钟离       | zhongli_(genshin_impact)     |         0.980978 |
| 65 | 烟绯       | yanfei_(genshin_impact)      |         0.980499 |
| 66 | 芙宁娜     | furina_(genshin_impact)      |         0.979196 |
| 67 | 重云       | chongyun_(genshin_impact)    |         0.972725 |
  
- Not Have

|    | name     | person   |   detected_name |
|---:|:---------|:---------|----------------:|
|  5 | 多莉     | {}       |             nan |
|  6 | 艾梅莉埃 | {}       |             nan |
| 18 | 玛拉妮   | {}       |             nan |
| 24 | 千织     | {}       |             nan |
| 29 | 坎蒂丝   | {}       |             nan |
| 32 | 埃洛伊   | {}       |             nan |
| 38 | 菲米尼   | {}       |             nan |
| 43 | 班尼特   | {}       |             nan |
| 47 | 白术     | {}       |             nan |
| 48 | 希格雯   | {}       |             nan |
| 52 | 夏沃蕾   | {}       |             nan |
| 56 | 嘉明     | {}       |             nan |
| 59 | 米卡     | {}       |             nan |
| 63 | 卡齐娜   | {}       |             nan |
| 67 | 瑶瑶     | {}       |             nan |
| 78 | 辛焱     | {}       |             nan |
| 79 | 赛索斯   | {}       |             nan |
| 84 | 夏洛蒂   | {}       |             nan |


## Supported or Developing Features

* [Tachie(差分) Detection and Clustering](https://github.com/deepghs/imgutils#tachie%E5%B7%AE%E5%88%86-detection-and-clustering)
* [Contrastive Character Image Pretraining](https://github.com/deepghs/imgutils#contrastive-character-image-pretraining)
* [Object Detection](https://github.com/deepghs/imgutils#object-detection)
* [Edge Detection / Lineart Generation](https://github.com/deepghs/imgutils#edge-detection--lineart-generation)
* [Monochrome Image Detection](https://github.com/deepghs/imgutils#monochrome-image-detection)
* [Truncated Image Check](https://github.com/deepghs/imgutils#truncated-image-check)
* [Image Tagging](https://github.com/deepghs/imgutils#image-tagging)
* [Character Extraction](https://github.com/deepghs/imgutils#character-extraction)

`imgutils` also includes many other features besides that. For detailed descriptions and examples, please refer to
the [official documentation](https://deepghs.github.io/imgutils/main/index.html). Here, we won't go into each of them
individually.

### Tachie(差分) Detection and Clustering

For the dataset, we need to filter the differences between the tachie(差分). As shown in the following picture

![tachie](https://deepghs.github.io/imgutils/main/_images/lpips_full.plot.py.svg)

We can use `lpips_clustering` to cluster such situations as shown below

```python
from imgutils.metrics import lpips_clustering

images = [f'lpips/{i}.jpg' for i in range(1, 10)]
print(images)
# ['lpips/1.jpg', 'lpips/2.jpg', 'lpips/3.jpg', 'lpips/4.jpg', 'lpips/5.jpg', 'lpips/6.jpg', 'lpips/7.jpg', 'lpips/8.jpg', 'lpips/9.jpg']
print(lpips_clustering(images))  # -1 means noises, the same as that in sklearn
# [0, 0, 0, 1, 1, -1, -1, -1, -1]
```

### Contrastive Character Image Pretraining

We can use `imgutils` to extract features from anime character images (containing only a single character), calculate
the visual dissimilarity between two characters, and determine whether two images depict the same character. We can also
perform clustering operations based on this metric, as shown below

![ccip](https://github.com/deepghs/imgutils/blob/gh-pages/main/_images/ccip_full.plot.py.svg)

```python
from imgutils.metrics import ccip_difference, ccip_clustering

# same character
print(ccip_difference('ccip/1.jpg', 'ccip/2.jpg'))  # 0.16583099961280823

# different characters
print(ccip_difference('ccip/1.jpg', 'ccip/6.jpg'))  # 0.42947039008140564
print(ccip_difference('ccip/1.jpg', 'ccip/7.jpg'))  # 0.4037521779537201
print(ccip_difference('ccip/2.jpg', 'ccip/6.jpg'))  # 0.4371533691883087
print(ccip_difference('ccip/2.jpg', 'ccip/7.jpg'))  # 0.40748104453086853
print(ccip_difference('ccip/6.jpg', 'ccip/7.jpg'))  # 0.392294704914093

images = [f'ccip/{i}.jpg' for i in range(1, 13)]
print(images)
# ['ccip/1.jpg', 'ccip/2.jpg', 'ccip/3.jpg', 'ccip/4.jpg', 'ccip/5.jpg', 'ccip/6.jpg', 'ccip/7.jpg', 'ccip/8.jpg', 'ccip/9.jpg', 'ccip/10.jpg', 'ccip/11.jpg', 'ccip/12.jpg']
print(ccip_clustering(images, min_samples=2))  # few images, min_sample should not be too large
# [0, 0, 0, 3, 3, 3, 1, 1, 1, 1, 2, 2]
```

For more usage, please refer
to [official documentation of CCIP](https://deepghs.github.io/imgutils/main/api_doc/metrics/ccip.html).

### Object Detection

Currently, object detection is supported for anime heads and person, as shown below

* Face Detection

![face detection](https://github.com/deepghs/imgutils/blob/gh-pages/main/_images/face_detect_demo.plot.py.svg)

* Head Detection

![head detection](https://github.com/deepghs/imgutils/blob/gh-pages/main/_images/head_detect_demo.plot.py.svg)

* Person Detection

![person detection](https://github.com/deepghs/imgutils/blob/gh-pages/main/_images/person_detect_demo.plot.py.svg)

Based on practical tests, head detection currently has a very stable performance and can be used for automation tasks.
However, person detection is still being further iterated and will focus on enhancing detection capabilities for
artistic illustrations in the future.

### Edge Detection / Lineart Generation

Anime images can be converted to line drawings using the model provided
by [patrickvonplaten/controlnet_aux](https://github.com/patrickvonplaten/controlnet_aux), as shown below.

![edge example](https://github.com/deepghs/imgutils/blob/gh-pages/main/_images/edge_demo.plot.py.svg)

It is worth noting that the `lineart` model may consume more computational resources, while `canny` is the fastest but
has average effect. Therefore, `lineart_anime` may be the most balanced choice in most cases.

### Monochrome Image Detection

When filtering the crawled images, we need to remove monochrome images. However, monochrome images are often not simply
composed of grayscale colors and may still contain colors, as shown by the first two rows of six images in the figure
below

![monochrome example](https://deepghs.github.io/imgutils/main/_images/monochrome.plot.py.svg)

We can use `is_monochrome` to determine whether an image is monochrome, as shown below:

```python
from imgutils.validate import is_monochrome

print(is_monochrome('mono/1.jpg'))  # monochrome images
# True
print(is_monochrome('mono/2.jpg'))
# True
print(is_monochrome('mono/3.jpg'))
# True
print(is_monochrome('mono/4.jpg'))
# True
print(is_monochrome('mono/5.jpg'))
# True
print(is_monochrome('mono/6.jpg'))
# True
print(is_monochrome('colored/7.jpg'))  # colored images
# False
print(is_monochrome('colored/8.jpg'))
# False
print(is_monochrome('colored/9.jpg'))
# False
print(is_monochrome('colored/10.jpg'))
# False
print(is_monochrome('colored/11.jpg'))
# False
print(is_monochrome('colored/12.jpg'))
# False
```

For more details, please refer to
the [official documentation](https://deepghs.github.io/imgutils/main/api_doc/validate/monochrome.html#module-imgutils.validate.monochrome)
.

### Truncated Image Check

The following code can be used to detect incomplete image files (such as images interrupted during the download
process):

```python
from imgutils.validate import is_truncated_file

if __name__ == '__main__':
    filename = 'test_jpg.jpg'
    if is_truncated_file(filename):
        print('This image is truncated, you\'d better '
              'remove this shit from your dataset.')
    else:
        print('This image is okay!')

```

### Image Tagging

The `imgutils` library integrates various anime-style image tagging models, allowing for results similar to the
following:

![tagging demo images](https://deepghs.github.io/imgutils/main/_images/tagging_demo.plot.py.svg)

The ratings, features, and characters in the image can be detected, like this:

```python
import os
from imgutils.tagging import get_wd14_tags

rating, features, chars = get_wd14_tags('skadi.jpg')
print(rating)
# {'general': 0.0011444687843322754, 'sensitive': 0.8876402974128723, 'questionable': 0.106781005859375, 'explicit': 0.000277101993560791}
print(features)
# {'1girl': 0.997527003288269, 'solo': 0.9797663688659668, 'long_hair': 0.9905703663825989, 'breasts': 0.9761719703674316,
#  'looking_at_viewer': 0.8981098532676697, 'bangs': 0.8810765743255615, 'large_breasts': 0.9498510360717773,
#  'shirt': 0.8377365469932556, 'red_eyes': 0.945058286190033, 'gloves': 0.9457170367240906, 'navel': 0.969594419002533,
#  'holding': 0.7881088852882385, 'hair_between_eyes': 0.7687551379203796, 'very_long_hair': 0.9301245212554932,
#  'standing': 0.6703325510025024, 'white_hair': 0.5292627811431885, 'short_sleeves': 0.8677047491073608,
#  'grey_hair': 0.5859264731407166, 'thighs': 0.9536856412887573, 'cowboy_shot': 0.8056888580322266,
#  'sweat': 0.8394746780395508, 'outdoors': 0.9473626613616943, 'parted_lips': 0.8986269235610962,
#  'sky': 0.9385137557983398, 'shorts': 0.8408567905426025, 'alternate_costume': 0.4245271384716034,
#  'day': 0.931140661239624, 'black_gloves': 0.8830795884132385, 'midriff': 0.7279844284057617,
#  'artist_name': 0.5333830714225769, 'cloud': 0.64717698097229, 'stomach': 0.9516432285308838,
#  'blue_sky': 0.9655293226242065, 'crop_top': 0.9485014081001282, 'black_shirt': 0.7366660833358765,
#  'short_shorts': 0.7161656618118286, 'ass_visible_through_thighs': 0.5858667492866516,
#  'black_shorts': 0.6186309456825256, 'thigh_gap': 0.41193312406539917, 'no_headwear': 0.467605859041214,
#  'low-tied_long_hair': 0.36282333731651306, 'sportswear': 0.3756745457649231, 'motion_blur': 0.5091936588287354,
#  'baseball_bat': 0.951993465423584, 'baseball': 0.5634750723838806, 'holding_baseball_bat': 0.8232709169387817}
print(chars)
# {'skadi_(arknights)': 0.9869340658187866}

rating, features, chars = get_wd14_tags('hutao.jpg')
print(rating)
# {'general': 0.49491602182388306, 'sensitive': 0.5193622708320618, 'questionable': 0.003406703472137451,
#  'explicit': 0.0007208287715911865}
print(features)
# {'1girl': 0.9798132181167603, 'solo': 0.8046203851699829, 'long_hair': 0.7596215009689331,
#  'looking_at_viewer': 0.7620116472244263, 'blush': 0.46084529161453247, 'smile': 0.48454540967941284,
#  'bangs': 0.5152207016944885, 'skirt': 0.8023070096969604, 'brown_hair': 0.8653596639633179,
#  'hair_ornament': 0.7201820611953735, 'red_eyes': 0.7816740870475769, 'long_sleeves': 0.697688639163971,
#  'twintails': 0.8974947333335876, 'school_uniform': 0.7491052746772766, 'jacket': 0.5015512704849243,
#  'flower': 0.6401398181915283, 'ahoge': 0.43420469760894775, 'pleated_skirt': 0.4528769850730896,
#  'outdoors': 0.5730487704277039, 'tongue': 0.6739872694015503, 'hair_flower': 0.5545973181724548,
#  'tongue_out': 0.6946243047714233, 'bag': 0.5487751364707947, 'symbol-shaped_pupils': 0.7439308166503906,
#  'blazer': 0.4186026453971863, 'backpack': 0.47378358244895935, ':p': 0.4690653085708618, 'ghost': 0.7565015554428101}
print(chars)
# {'hu_tao_(genshin_impact)': 0.9262397289276123, 'boo_tao_(genshin_impact)': 0.942080020904541}
```

We currently integrate the following tagging models:

* [Deepdanbooru model](https://deepghs.github.io/imgutils/main/api_doc/tagging/deepdanbooru.html), but not recommended
  for production use.
* [wd14-v2 model](https://deepghs.github.io/imgutils/main/api_doc/tagging/wd14.html#), inspired
  by [SmilingWolf/wd-v1-4-tags](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags).

In addition, if you need to convert the dict-formatted data mentioned above into the text format required for image
training and tagging, you can also use the `tags_to_text` function (see the
link [here](https://deepghs.github.io/imgutils/main/api_doc/tagging/format.html#tags-to-text)) for formatting, as shown
below:

```python
from imgutils.tagging import tags_to_text

# a group of tags
tags = {
    'panty_pull': 0.6826801300048828,
    'panties': 0.958938717842102,
    'drinking_glass': 0.9340789318084717,
    'areola_slip': 0.41196826100349426,
    '1girl': 0.9988248348236084
}

print(tags_to_text(tags))
# '1girl, panties, drinking_glass, panty_pull, areola_slip'
print(tags_to_text(tags, use_spaces=True))
# '1girl, panties, drinking glass, panty pull, areola slip'
print(tags_to_text(tags, include_score=True))
# '(1girl:0.999), (panties:0.959), (drinking_glass:0.934), (panty_pull:0.683), (areola_slip:0.412)'
```

### Character Extraction

When we need to extract the character parts from anime images, we can use
the [`segment-rgba-with-isnetis`](https://deepghs.github.io/imgutils/main/api_doc/segment/isnetis.html#segment-rgba-with-isnetis)
function for extraction and obtain an RGBA format image (with the background part being transparent), just like the
example shown below.

![isnetis](https://deepghs.github.io/imgutils/main/_images/isnetis_trans.plot.py.svg)

```python
from imgutils.segment import segment_rgba_with_isnetis

mask_, image_ = segment_rgba_with_isnetis('hutao.png')
image_.save('hutao_seg.png')

mask_, image_ = segment_rgba_with_isnetis('skadi.jpg')
image_.save('skadi_seg.png')
```

This model can be found at https://huggingface.co/skytnt/anime-seg .

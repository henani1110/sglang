# 600 样本质量集：授权与来源审计

**状态：未冻结。** 逐层区分数据集问答标注/评测脚本的 license 与图片/文档媒体本身的来源和使用条件。公开可下载或 benchmark 常用，不等于每张图片均可再分发。质量集只在本地与授权 SSD 上用于研究；归档默认保存 sample ID、来源 URL、许可证文字/URL、SHA-256 和评分，不把图像打入对外包。

| 子集 | 标注/脚本来源 | 媒体来源 | 当前证据 | 冻结前处理 |
|---|---|---|---|---|
| POPE 120 | [官方论文](https://arxiv.org/abs/2305.10355)提供代码与数据；[作者仓库](https://github.com/RUCAIBox/POPE)的仓库许可为 MIT。 | POPE 构造示例使用 COCO 图片。COCO 官方 terms 说 annotation 为 CC BY 4.0，但 Consortium 不拥有图片版权，图片使用需遵守 Flickr terms。 | 不能把 POPE 仓库 MIT 或 COCO annotation CC BY 误认为图片整体 CC BY。 | 按 image ID/原始 Flickr URL 获取逐图许可元数据；只抽到有明确研究使用基础的样本。记录 COCO license ID 与源链接；无法确认则替换并记录。 |
| GQA 120 | [GQA 官方下载页](https://cs.stanford.edu/people/dorarad/gqa/download.html)提供问答/图像数据；页面未显式给出整体 license。 | GQA 基于 Visual Genome/COCO 图片；Visual Genome API 提供 image id、URL、COCO/Flickr ID。 | 上游图片权利需依实际源图判断。 | 保存 GQA 问题 ID、VG image ID、来源 URL、source ID、授权证据。只选择可确认媒体来源与研究条件的样本；不把 GQA 下载可得性当成许可。 |
| TextVQA 120 | [TextVQA 官方数据页](https://textvqa.org/dataset/)明确标记数据 CC BY 4.0。 | 官方页称图像来自 OpenImages train/test；Open Images 原始论文说明图像有 Creative Commons Attribution 许可。 | 此五项中公开授权表述最清楚；逐样本仍保留 OpenImages ID 与归属。 | 使用官方 v0.5.1 split；按官方图像方向字段纠正；记录数据文件版本、图片 ID、source URL、许可版本和 SHA-256。 |
| DocVQA 120 | [官方数据页](https://site.docvqa.org/datasets/docvqa)要求经 RRC 门户登录下载，并明确要求先阅读下载页面的 terms and conditions。 | 官方页称图片来自 UCSF Industry Documents Library。UCSF IDL FAQ 称非商业项目在符合 fair use 时可用；这不是一个覆盖所有底层文件的 blanket license。 | 当前没有读到 RRC 下载页实际 terms，也未确认每个底层文档的版权状态。 | 不把镜像仓库的 MIT/Apache 标签当作上游文档授权。先通过正式 RRC 条款确认研究评测可用范围，并逐 item 记录 IDL record/terms；有歧义的样本替换或仅作可识别的未冻结项。 |
| ChartQA 120 | [作者仓库](https://github.com/vis-nlp/ChartQA)标示 GPL-3.0，但这只能证明仓库/对应文件许可，不能自动证明外部图表图片的权利。 | 论文/仓库中的 chart 图片来自网页/报告；不同来源的再使用条件可能不同。 | 不能将 GPL-3.0 直接外推到所有图像。 | 保留 ChartQA image ID、源网页、来源组织、网页/图片许可、抓取日期和 SHA-256。只选有明确许可/开放数据条件的图表；不能确认则替换或在报告中明确无法入选，不把不明媒体称“已许可”。 |

## 数据冻结规程

1. 按每个 benchmark 的预先声明 split 和官方 QA/score 格式抽取候选项；每项是 image-question-answer 记录。
2. 采用固定抽样 seed `20261109`，按候选序列逐项审查媒体授权；未通过项目在 `excluded.csv` 记录来源、SHA、拒绝理由与替代候选，不覆盖旧行。
3. 为每个最终项目记录：`dataset`, `split`, `question_id`, `image_id`, `image_sha256`, `annotation_license_url`, `media_source_url`, `media_license_name`, `media_license_url`, `permission_evidence`, `attribution`, `eligible_for_local_research`, `redistributable`, `qa_sha256`, `prompt_sha256`。
4. 图片不随研究归档对外分发；本地和 SSD 副本逐项 SHA-256 校验。许可清楚但不允许再分发的图像仍只能放在受控本地/SSD；公开报告只保留 ID、哈希和官方引用。
5. 每个 benchmark 最终必须正好 120 条可用于本地评测的配对样本；若无法凑足，不伪称达到 600 个已许可样本，继续核查正式许可、原始数据门户条款和可替代来源。此状态不停止 Goal 的其他独立工作。

## 主要来源

- [POPE official paper](https://arxiv.org/abs/2305.10355), [official repository](https://github.com/RUCAIBox/POPE)
- [COCO official terms](https://cocodataset.org/#termsofuse)；[terms text](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)
- [GQA official download](https://cs.stanford.edu/people/dorarad/gqa/download.html)；[Visual Genome source metadata](https://visualgenome.org/api/v0/api_readme)
- [TextVQA official dataset page](https://textvqa.org/dataset/)
- [Open Images paper](https://arxiv.org/abs/1811.00982)
- [DocVQA official dataset page](https://site.docvqa.org/datasets/docvqa)；[UCSF IDL FAQ](https://idl.ucsf.edu/chemical/help/faq/)；[UCSF IDL collection policy](https://www.industrydocuments.ucsf.edu/wp-content/uploads/2022/05/IDL_Collection_Development_Policy_2022.pdf)
- [ChartQA author repository and README](https://github.com/vis-nlp/ChartQA)

## 2026-11-09 官方页面复核补充

- GQA 官方主页明确说其图片来自 COCO 和 Flickr，scene graph 基于 Visual Genome；官方下载页列出问答、图片和 scene graphs，但未展示覆盖所有图像的总许可。不能仅因为 GQA 自托管下载就将它登记成统一 CC license。
- COCO 官方 terms 将 annotations/website 与 images 分开：annotations 为 CC BY 4.0；COCO Consortium 声明不拥有图片版权，图片按 Flickr terms 使用，使用者承担复制图片的责任。COCO JSON 中 per-image license 字段可以辅助核验，但具体 URL、当前 source page 与允许用途仍需留存。
- TextVQA 官方数据页明确标注数据 CC BY 4.0，原始论文称图片来自 Open Images train/test；Open Images 论文描述其图像采用 Creative Commons Attribution license。抽样 manifest 仍记录 Open Images source ID 和归属。
- DocVQA 官方数据说明图像来自 UCSF Industry Documents Library，下载须经 RRC challenge portal；AWS Registry 的数据条目将许可链接到 UCSF copyright page。这个来源链没有自动消除每份 IDL 文件的限制，RRC 下载页 terms 和实际使用范围仍要核实。
- ChartQA 作者仓库显示 GPL-3.0 的 repository license，并提供图像文件；该许可不能当然覆盖外部来源图表。官方论文提到其图表来自 Statista、Pew Research、Our World in Data、OECD；OWID 官方 FAQ 要求复用内容遵守 underlying third-party provider terms 并正确引用。因此需逐图登记源站及许可，当前仍不宣称其 120 张图片全部清权。

以上为官方页面事实及其适用边界，不构成对任何单张图片使用权的法律判断。

## 2026-11-09 第二轮官方来源与平台条款核对

- `[事实 / COCO 与 Flickr]` COCO 官方条款将 annotations/website 标为 CC BY 4.0，但声明 COCO Consortium 不拥有图片版权，图片须遵循 Flickr Terms，图片复制责任由使用者承担（[COCO Terms](https://cocodataset.org/dataset/termsofuse.htm)）。Flickr API 官方条款进一步明确：照片由用户/摄影者拥有，API 不覆盖作者对单图施加的限制，包括 All Rights Reserved、CC 或其他约定（[Flickr API Terms](https://www.flickr.com/help/terms/api)）。当前 Flickr Terms 最近修订为 2025-07-24，并限制未经授权的收集、复制、下载和数据挖掘（[Flickr Terms](https://www.flickr.com/help/terms)）。因此不以公开 URL 可访问或 COCO annotations CC BY 为单图许可，不直接批量抓取 Flickr；只有逐图许可证据和合规来源路径齐备后才纳入。
- `[事实 / POPE]` POPE 当前官方仓库有 MIT `LICENSE`，文本授权范围写作 Software。此证据支持仓库代码许可，不单独证明所有 COCO 图像或 POPE 标注/生成文件的再分发权（[POPE LICENSE](https://github.com/RUCAIBox/POPE/blob/main/LICENSE)）。样本 manifest 需将代码与每个图片媒体授权分开。
- `[事实 / GQA 与 Visual Genome]` GQA 官方主页明确说明图片来自 COCO 与 Flickr，scene graphs 基于 Visual Genome；官方下载页提供 images/questions，但未见适用于全部图片的统一 license。Visual Genome image metadata 提供 `coco_id`、`flickr_id` 和源 URL，可用于逐项回溯，但本身不授予图像版权（[GQA home](https://cs.stanford.edu/people/dorarad/gqa/)、[GQA download](https://cs.stanford.edu/people/dorarad/gqa/download.html)、[Visual Genome metadata API](https://visualgenome.org/api/v0/api_readme)）。
- `[事实 / TextVQA 与 Open Images]` TextVQA 官方页面将 v0.5.1 data 标为 CC BY 4.0，并注明图片来自 Open Images train/test（[TextVQA dataset](https://textvqa.org/dataset/)）。Open Images V3 官方 README 表示其 image metadata 含原始 landing URL、license、作者等字段；同时声明“listed as CC BY 2.0”不构成逐图担保，使用者仍应核验每张图（[Open Images V3 README](https://github.com/openimages/dataset/blob/main/READMEV3.md)）。因此这是五类中最适合先筛选的一类，但只有取得匹配的 V3 image row、核对 license/author/source、保存 SHA 和合规下载来源后才记为 eligible。
- `[事实 / DocVQA]` 官方 DocVQA 页面确认图片取自 UCSF Industry Documents Library，并要求从 RRC challenge 下载、先阅读下载页条款（[DocVQA dataset](https://www.docvqa.org/datasets/docvqa)）。本轮 web 工具未能读取 RRC 下载页，且该页需登录；在账户条款可见并核对前，DocVQA 候选不计为 eligible。
- `[事实 / ChartQA]` 作者仓库 GPL-3.0 仅作为仓库标记，不能自动覆盖网页或报告中的图表图片；ChartQA 图像仍逐图追溯源网页及媒体授权。当前没有新增已清权图像。
- `[决定 / 样本处理]` 为后续逐项检查建立空白 `datasets/rights_manifest.csv` 与 `datasets/excluded.csv`。当前 eligible 样本数仍为 0/600；不得以此停掉其它独立实验。先从 TextVQA/Open Images v3 的许可 metadata 交叉匹配开始，POPE/GQA 只纳入能够从 benchmark ID 回溯到明确 CC/公有领域授权的图像，DocVQA 等待正式 RRC 条款访问。

以上是对公开条款与来源说明的工作记录，不构成针对某张图片的法律意见；逐项资格判断保留在 manifest，含糊项不计入已许可样本。

## 2026-11-09 TextVQA/Open Images V3 候选清单（未清权）

- `[事实 / 数据源]` SSD 保存的 TextVQA v0.5.1 train/val JSON SHA-256 分别为 `95f5c407db56cba56a177799dcd685a7cc0ec7c0d851b59910acf7786d31b68a` 与 `4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d`；Open Images V3 2017-11 官方 `images.csv` 元数据归档 SHA-256 为 `41aa0954834929ebf35c58b73e3a68f9a7a43fd08407d467b90aa2f883564e5d`。TextVQA train/val 共 39,602 个问题、25,119 个唯一图像 ID；所选 manifest 为固定 seed `20261109` 下 80 train calibration + 40 val heldout，120 个不同 image ID。
- `[事实 / 元数据匹配]` 这 120 个候选都在 Open Images V3 per-image metadata 中匹配 `CC BY 2.0`、author 和原始 landing URL。Open Images README 明确说明 metadata 的 license 标记不作保证，仍须核验原始源页面；因此候选 CSV 中 120 项的 `eligible_for_local_research` 和 `redistributable` 均保持 pending，`image_sha256` 与 rotation 仍空，状态均为 `candidate_license_metadata_matched_not_cleared`。
- `[事实 / 当前计数]` 原始 Flickr 图片页核验数为 0/120，eligible 为 0/120，下载图像为 0。Flickr 现行 Terms 限制未经授权的批量复制/下载；本阶段不通过脚本抓取图片。即使未来可本地研究，也需另行记录再分发许可，不能从 CC 标记自动推定再分发权。
- `[决定]` TextVQA 只作为优先候选池继续检查，不计入已许可 600 项、不冻结质量集。保留 `textvqa-120-candidates-v0.1.csv` 与摘要 JSON；若源页和合规获取路径无法逐项确认，就标记 exclusion 并从其他允许来源继续补充。此状态不阻止无图合成机制校验、已授权数据工作及计时探针继续推进。

## 2026-11-09 官方数据页面字段与来源路径复核

- `[事实 / COCO]` COCO 官方 Terms 将 annotation/website 标为 CC BY 4.0，同时明确 Consortium 不拥有图片版权、图片需遵循 Flickr Terms。对 POPE 的可行筛选路径是使用其 COCO 图像 ID 回连 COCO 官方 image record，再按每图许可与合规来源核查；POPE repo 的 MIT 不能替代图片授权。
- `[事实 / GQA]` GQA 官方主页明确列明图片来自 COCO 与 Flickr。后续只保留能用 GQA/VG ID 回到 COCO image record、且通过逐图媒体权利审查的项目；Flickr-only 项不由 GQA 问答或 Visual Genome API 自动获得许可。
- `[事实 / TextVQA]` TextVQA 官方 v0.5.1 页将其 data 标记 CC BY 4.0，并在每条样本 schema 中提供 `image_id`、`flickr_original_url`、`flickr_300k_url`。Open Images V3 metadata 提供 OriginalLandingURL、License、Author 等字段，但官方 README 明确不担保逐图 license status，要求使用者自行核验。现有 120 项候选仍是 source-page pending：0/120 confirmed、0/120 eligible、0 张本地图片；本轮未对 Flickr 执行批量抓取或下载。
- `[事实 / Open Images rotation]` Open Images metadata 文档列出图像方向/Rotation 字段；TextVQA 官方也提示数据中有旋转图像。当前 TextVQA 候选来自 V3 metadata 包，该表没有提供可直接用于这 120 项的 rotation 值，因此不能把空字段当作 0°。下一批候选需从带 rotation 的官方元数据交叉核验，或在取得合规本地副本后另行确认方向。
- `[事实 / ChartQA 与 DocVQA]` ChartQA 作者仓库显示 GPL-3.0 repository license 并分发 PNG/QA/table 文件，但本轮未在官方 README 找到覆盖所有图表来源图片的统一媒体许可。DocVQA 官方页仍说明图片来自 UCSF Industry Documents Library、经 RRC 页面下载；该页没有提供可替代 RRC 下载条款的图片许可声明。
- `[推断 / 收敛]` 五类固定 benchmark 中，目前不能用 repository/dataset license 一步冻结任何图片。POPE/GQA 可尝试通过 COCO per-image license ID 缩小候选池；TextVQA 需 source-page + rotation；ChartQA 需逐图 source publisher；DocVQA 需 RRC 条款与底层 record 范围。保持 0 eligible 比放宽判据更可复现。
- `[决定 / 后续筛选]` 将抽样顺序调整为：先筛 COCO-backed POPE/GQA 的逐图明确许可记录，再继续 TextVQA source/rotation 核查，然后判定 ChartQA/DocVQA 是否存在可合法使用的子集；每个数据集仍以 120 个真正通过项目为目标，任何不足都保留 gap 并继续寻找授权替代来源，不把 pending 项计数为通过。

主要官方页面：[COCO Terms](https://cocodataset.org/dataset/termsofuse.htm)、[GQA home](https://cs.stanford.edu/people/dorarad/gqa/index.html)、[TextVQA dataset](https://textvqa.org/dataset/)、[Open Images V3 README](https://github.com/openimages/dataset/blob/main/READMEV3.md)、[Open Images rotation metadata](https://storage.googleapis.com/openimages/web/download.html)、[ChartQA repository](https://github.com/vis-nlp/ChartQA)、[DocVQA dataset](https://site.docvqa.org/datasets/docvqa)。

## 2026-11-09 POPE/COCO 逐图 license metadata 候选（仍未清权）

- `[事实 / 官方输入]` COCO 官方 `annotations_trainval2014.zip`（252,872,794 bytes）已存于 SSD 与本地，SHA-256 `031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009`；只提取了 CC BY 4.0 的 `instances_val2014.json`（SHA-256 `e1b4d1bc54906827b1a029981b1c77f2088bd4462569b7615db7929070c3ab47`），未下载任何 COCO/Flickr 图像。POPE 作者仓库固定 commit 记录在 `datasets/source/pope/upstream-commit.txt`，三个官方 question files 各 3,000 条，全部 9,000 条均能与 COCO val2014 `image_id` 匹配。
- `[事实 / metadata 候选池]` 三类 POPE 文件的图像集合一致：COCO `license_id=4` 有 74 个 image IDs、`license_id=5` 有 51 个；两者分别对应 CC BY 2.0、CC BY-SA 2.0。按 `license_ids=[4,5]` 得到三类共有 125 个 metadata candidates，固定 seed `20261109` 抽取 random/popular/adversarial 各 40 条，三组 image ID 互不重叠，总计 120 question-image rows / 120 unique image IDs。所选记录中 license ID 4 为 71 项、ID 5 为 49 项。
- `[事实 / annotation license 边界]` POPE 官方仓库的 `LICENSE` 标为 MIT，但其授权文字以 “Software” 定义许可对象；官方 README 说明 repo 提供 source code/data，但本轮未找到单独说明 JSON question/label data 的 license。候选 manifest 不把 MIT 直接登记为 Q/A 数据授权，`annotation_license_url` 留空，逐项状态为 `pending_annotation_and_source_page_review`。
- `[事实 / 当前计数]` POPE v0.5 candidate manifest 已记录 question-record、COCO image-record、prompt 和源包 hashes、Flickr 原始 URL、COCO 图像 CDN URL 与 license ID；Flickr source-page 核验 0/120，annotation permission 确认 0/120，eligible 0/120，图片下载 0。v0.1–v0.4 的抽样与更正版本保留；v0.5 是显式区分 metadata candidate 与 eligible 状态的版本。
- `[决定]` v0.5 只用于后续逐图清权与问答许可查询，不进入 `rights_manifest.csv` 的已许可项，不算作已授权 120。CC BY-SA 样本的 attribution/ShareAlike 与再分发权限继续分别审查；没有 source-page/annotation evidence 以前不使用图片、不报告质量分数。

> GQA 的 image metadata 下一步通过作者当前大学站点的 Visual Genome 图像映射文件回连 COCO ID。Visual Genome 原域当前网页被无关博彩内容污染，故不从该域下载，也不采用其当前页面声明。


## 2026-11-09 GQA/COCO metadata 候选（未清权）

- `[事实 / 官方输入]` 从 [GQA 官方下载页](https://cs.stanford.edu/people/dorarad/gqa/download.html)提供的 `questions1.2.zip` 中仅提取 `val_balanced_questions.json`，没有提取或下载任何 benchmark 图片。压缩包 SHA-256 为 `cffe92775624693799d70c15db631d4373e4d09b44153e1757ccf1d5fb3fc623`，平衡验证问答 JSON SHA-256 为 `2f675643c0cfffe0485dbae0c0c0dfc16b547443309a98544502bc1ae87c0779`。
- `[事实 / 映射]` 使用 Visual Genome 作者的 [University of Washington image metadata 页面](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)及 `image_data.json.zip`（SHA-256 `b87a94918cb2ff4d952cf1dfeca0b9cf6cd6fd204c2f8704645653be1163681a`），将 GQA `imageId` 映射至 Visual Genome `image_id`，再按 `coco_id` 关联 COCO `instances_train2014.json` 与 `instances_val2014.json`。COCO archive SHA-256 为 `031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009`；train/val instance JSON 的 SHA 分别为 `777ef504abfeba349930485bb32ecd5b6529129a3e010f7852e1437cb049b732` 与 `e1b4d1bc54906827b1a029981b1c77f2088bd4462569b7615db7929070c3ab47`。
- `[观察 / 候选抽取]` 固定 seed `20261109` 扫描 GQA balanced validation 的 132,062 条问答。67,808 条没有 Visual Genome ID 映射；在成功映射的记录中没有 COCO ID 缺失。按 COCO 每图 metadata license ID 4/5 过滤后有 1,326 个不同 COCO image candidates；确定性抽出 120 个不同图片 ID，license metadata ID 4/5 分布为 80/40。Visual Genome metadata 有 290 条重复 COCO ID 记录；抽样键为 COCO ID，候选按唯一媒体 ID 去重。
- `[事实 / 权利状态]` `gqa-120-candidates-v0.1.csv` 和摘要 JSON 保存 GQA question/image、Visual Genome image、COCO image-record、prompt hashes、原始来源 URL 及 license metadata。GQA 官方下载页没有列出整体 Q/A annotation license；COCO 条款要求图片遵循原始 Flickr/source 条件，metadata license ID 不能替代当前源页面确认。因此 annotation permission 0/120、source page 0/120、eligible 0/120、图片下载 0；120 条均为 pending candidates，不进入冻结质量集。
- `[决定]` 保留 GQA 候选供后续逐项查询问答数据授权与媒体源页面；在两类证据和合规取得路径确认前不读取或下载图像、不做答案质量评分。source archive、提取 JSON、COCO/Visual Genome metadata、候选脚本和清单均留有本地与 SSD 副本及 SHA-256。


## 2026-11-09 GQA annotation-license 主张冲突复核

- [事实 / 新线索] GQA 官方下载页的可检索正文没有机器可读的 license 文字，但 Salesforce AI Research 的 [README](https://github.com/SalesforceAIResearch/visual-unit-testing) 直接转引称 Stanford 下载页底部有 CC-BY 4.0 badge，并提醒 GQA 图片来自 COCO/Flickr、保留逐图媒体条款。NeurIPS 2025 和 2023 的补充材料也将 GQA 标为 CC BY 4.0（[2025 paper](https://openreview.net/pdf?id=KXmDTGKwhy), [2023 supplement](https://proceedings.neurips.cc/paper_files/paper/2023/file/6a42b45af2b72e6e5b3a6fe695809f-Supplemental-Datasets_and_Benchmarks.pdf)）。
- [事实 / 冲突] Google DeepMind ProEval data README 在其 benchmark exception list 中另将 GQA 标为 MIT（[README](https://github.com/google-deepmind/proeval/blob/main/data/README.md)）；该文件同时说明 repository 不是官方 Google 产品，也不是 GQA 原作者的许可文件。这个下游标注与 CC BY 4.0 的引用不一致。
- [推断 / 权利状态] 当前证据使 CC BY 4.0 成为较强的 GQA annotation-license 主张，但原始 license badge 对 questions1.2.zip 的适用范围和图像媒体边界还没有从原始 grant 文本逐字确认，并有下游 MIT 冲突。因此把状态细化为 reported CC BY 4.0; grant scope pending primary confirmation；候选 120 条仍是 annotation/source-page/eligibility pending，0/120 eligible，不把下游陈述当成最终授权。
- [决定] 保留上述对立证据并查找可保存的官方 badge/artifact 或作者发布的许可说明；即使 GQA 问答标注许可闭合，也继续逐图区分 COCO/Flickr 原图媒体权利。详细材料见 datasets/manifests/gqa-license-evidence-review-v0.1.md。


## 2026-11-09 直接官方页面复核：许可对象与来源链

- `[事实 / POPE]` 直接阅读 [POPE 官方 LICENSE](https://github.com/RUCAIBox/POPE/blob/main/LICENSE)：MIT 文本把许可对象定义为 “Software and associated documentation files (the "Software")”。仓库含 POPE question JSON 不足以证明该 MIT grant 覆盖这些标注数据；继续将 question/label permission 记为 pending。COCO image rights 仍按逐图 Flickr/source terms 独立审查。
- `[事实 / GQA]` 当前 [GQA 官方下载页](https://cs.stanford.edu/people/dorarad/gqa/download.html)列出 question v1.2、scene graph v1.1 与图片下载入口及 JSON schema；可见页面没有列出适用于问题/答案文件的 license 文本。[GQA 官方主页](https://cs.stanford.edu/people/dorarad/gqa/)明确图片来自 COCO 和 Flickr。现有 CC-BY-4.0 badge 转述线索仍不能替代原始 grant 的适用范围核验。
- `[事实 / DocVQA]` 当前 [DocVQA 官方数据页](https://site.docvqa.org/datasets)要求在 RRC challenge portal 登录下载，并要求先阅读 Download 页的 terms and conditions；该页还说明 Task 1 文档页面来自 UCSF Industry Documents Library，并在 schema 中提供 `ucsf_document_id`。未登录获取并存档实际条款前，annotation 与每份底层文档的使用条件都保持 pending。
- `[事实 / TextVQA/Open Images]` [TextVQA 官方数据页](https://textvqa.org/dataset/)的官方索引正文仍标为 CC BY 4.0，并称 v0.5.1 的图片来自 Open Images；本次页面直开只返回 JavaScript shell，故正文证据来自该官方页面的索引内容。[Open Images V6 官方 metadata说明](https://storage.googleapis.com/openimages/web/download.html)包含 `OriginalLandingURL`、License、Author、Title 和 rotation；其 V4 页面同时提醒提供方不保证各图片的 license 状态，使用者应自行核验。因此 TextVQA 页面许可标注不直接替代每张底层 Flickr 图片的 source landing page、license 与 attribution 记录。
- `[事实 / ChartQA]` [ChartQA 官方仓库 LICENSE](https://github.com/vis-nlp/ChartQA/blob/main/LICENSE)为 GPL-3.0；[官方 README](https://github.com/vis-nlp/ChartQA/blob/main/README.md)说明仓库/Hugging Face 有 chart 图片与数据，且 Pew Research Centre 的部分图表由作者手工标注。repo 级 GPL 文件并未逐项陈述外部图表媒体的授权，故当前仍不将它登记为 120 张图片的统一 media license。
- `[观察 / 当前结论]` 这些直接核对没有把任何现有候选升级为 eligible；`rights_manifest.csv` 仍空，质量集 eligible 仍为 0/600。POPE/GQA 的 source URL 与 metadata 级候选可继续供核查，不能进入图像质量评分。本文记录的是来源页能直接支持的范围，不对授权范围作扩张解释。
- `[待办]` 继续寻找 POPE/GQA 原始 annotation grant；按 Open Images 原始 metadata 回查 TextVQA 的 OpenImages ID 与 landing page；登录 RRC 后逐条存档 DocVQA 条款并按 UCSF document ID 复核；为 ChartQA 候选逐图追溯源网页及媒体授权。不能闭合的候选须留在 pending/excluded 记录，不能为凑齐 120 而推断授权。

- `[事实 / 逐图访问尝试]` 2026-11-09 用 Web 直接打开 TextVQA 候选前 5 张图的 Flickr landing URL 均由工具返回页面不可访问/内部错误；页面正文获取数 0，授权核验数 0。尝试与结果记在 `datasets/review/textvqa-source-page-access-attempts-20261109.csv` 和同名 JSON；候选状态仍 pending。


## 2026-11-09 Flickr automated-page-check method review

- `[事实 / 官方条款]` 本轮在执行逐页校验后复核了 [Flickr Terms of Use](https://www.flickr.com/help/terms/)：其 Restrictions 明确将 scraping/data mining/robots 类采集列为禁止行为，并限制下载 Flickr Materials 与其中信息。官方 [Flickr API 文档](https://www.flickr.com/services/api/)说明 API 需应用 key，API 使用另受 [Flickr APIs Terms of Use](https://www.flickr.com/help/terms/api)约束。
- `[事实 / 已发生的尝试]` 远端脚本曾对 TextVQA 120 个 Flickr landing pages 依序发起 HTML GET，仅解析结构化 license/作者/title 字段，没有下载图像；120 项中 HTTP 200 为 102，读到 license 且与 Open Images 元数据一致为 100，不一致为 2，页面错误/不可达为 18。原始提取字段及页面哈希保存在 `datasets/review/textvqa-flickr-source-page-audit-full-20261109.jsonl` 和 summary。该尝试未使用 Flickr API key。
- `[决定 / 证据效力]` 在核对到上述条款后，停止后续 Flickr 直接页面自动请求；将这批自动化 page results 仅作为过程审计保留，不作为合规清权或 sample eligibility 证据。所有 120 个 TextVQA 候选仍为 pending；`rights_manifest.csv` 仍空、eligible 仍为 0/600。没有下载图片，也没有启动基于这些页面结果的质量评分。
- `[待办 / 合规路径]` 若继续使用 Flickr source-page 元数据，需要取得并按官方 API 条款使用应用 key，或取得 Flickr/内容权利人的明确许可；否则只将 Open Images V3 逐图元数据和 source URL 作为候选线索，不能把它升级成 source-page confirmation。可独立继续 synthetic 系统实验、代码正确性和已获授权来源的研究。

## 2026-11-09 Open Images V3 与 Flickr 合规获取路径补充

- `[事实 / Open Images V3 元数据]` Open Images V3 官方 README 描述其 image list 字段包括 `OriginalLandingURL`、`License`、`Author`、`Title`、`OriginalMD5` 和 `Rotation`，示例是 Flickr source。README 也将图片列为 CC BY 2.0，同时明确 Open Images 不对每张图片的许可状态作保证，并要求使用者自行核验。该 V3 metadata row 可作来源/作者/许可证候选证据，不能替代当前来源页的确认；目前没有在本地 E 或已检查的 Win10 SSD 路径找到完整 `open-images-v3.tar.gz`，候选仍为 120 条、eligible 仍为 0。来源：[Open Images V3 README](https://github.com/openimages/dataset/blob/main/READMEV3.md)、[Open Images V6 下载说明中的图像信息字段](https://storage.googleapis.com/openimages/web/download.html)。
- `[事实 / Flickr API]` Flickr 官方 API 页面说明 API 使用须申请 application key；API Terms 仍要求遵守照片所有者逐图施加的限制，API 本身不改变图片许可，也不自动提供缓存/长期存储许可。故 API 只可能是一个受条款约束的取证路径，不会自动将任何候选改为 eligible。来源：[Flickr API overview](https://www.flickr.com/services/api/)、[Flickr API Terms](https://www.flickr.com/help/terms/api)、[Flickr API keys](https://www.flickr.com/services/api/misc.api_keys.html)。
- `[决定 / 数据状态]` 不再对 Flickr landing pages 运行脚本或进行批量下载。此前 120 页自动 GET 的 JSONL/摘要/异常表继续作为过程记录保留，但不作为许可核验、授权数据或样本筛选依据；不得因为 100 条观察到 license 字符串匹配就恢复资格。要继续使用 Flickr 派生样本，需先走官方 API/明确授权并保留许可来源；替代是使用权利链更直接的开放媒体或全合成、可自建标注集，并清楚标明它不是官方 benchmark 替代集。

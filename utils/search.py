# -*- coding: utf-8 -*-  # 指定文件编码为UTF-8，支持中文
"""
Time-sliced PubMed Review Search with LLM query generation.  # 文件说明：按时间切片的综述检索，含LLM生成检索式
Only searches and ranks review articles; does NOT fetch full text.  # 仅检索和排序综述，不获取全文
Requires: metapub, your api.generate_text (LLM).  # 依赖metapub与你的LLM接口api.generate_text
Optional: set env NCBI_API_KEY to improve E-utilities rate limits.  # 可设置NCBI_API_KEY提升速率限制
"""

import logging  # 日志记录
import math  # 数学函数库，用于sqrt、tanh等
import time  # 时间与sleep
from collections import defaultdict  # 提供默认字典结构
from datetime import datetime  # 获取当前年份等
from typing import Callable, Dict, List, Optional, Tuple  # 类型注解

from tqdm import tqdm  # 进度条显示

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from metapub import PubMedFetcher  # 引入PubMed检索器

fetch = PubMedFetcher()  # 创建全局fetch实例（内部有缓存）
# from api import generate_text  # 动态导入你的LLM接口
# -------------------- 1) LLM 生成 PubMed 检索式 --------------------
def llm_query_from_user_question(user_query: str) -> str:  # 将自然语言问题转为PubMed检索式
    """
    用 generate_text 把自然语言问题转成 PubMed 检索式（仅返回检索式）。  #
    不要在这里加日期，日期交给构造函数用 [dp] 做切片。  # 日期范围由后续build_query添加
    """
    
    prompt = f"""  
            你是资深生物医学信息检索专家。请为下述研究问题生成 PubMed 检索式（Boolean + MeSH + 自由词）：
            问题：{user_query}
            要求：
            1) 同时使用 MeSH 与自由词（Title/Abstract 用 [tiab]）
            2) 使用布尔运算符 AND/OR/NOT
            3) 限定文献类型为综述（Review[Publication Type]），但不要写日期范围
            4) 只输出检索式本身，不要任何解释、标点修饰或代码块
                """.strip()  # 去掉前后空白
    q = str(generate_text(prompt)).strip()  # 调LLM生成检索式并去空白
    for bad in ("```", "“", "”"):  # 清理常见包裹符号（代码块/中文引号）
        q = q.replace(bad, "")  # 替换为无
    return q  # 返回检索式字符串

# -------------------- 2) 构造检索式（叠加类型+日期+语言） --------------------
def build_query(base_query: str, y_from: int, y_to: int,  # 构造最终检索式
                strong_review: bool = False,  # 是否使用更强的综述类型集合
                lang_filter: Optional[List[str]] = None) -> str:  # 语言过滤列表
    review_clause = '(Review[Publication Type])'  # 基本综述限定
    if strong_review:  # 若要求更强综述
        review_clause = '(' + ' OR '.join([  # 合并Review/Systematic/Meta-Analysis三类
            'Review[Publication Type]',
            'Systematic Review[Publication Type]',
            'Meta-Analysis[Publication Type]'
        ]) + ')'  # 结束括号
    date_clause = f'("{y_from}/01/01"[dp] : "{y_to}/12/31"[dp])'  # 出版日期区间[dp]，按年切片
    lang_clause = ''  # 默认无语言约束
    if lang_filter:  # 如果传入语言过滤
        langs = ' OR '.join([f'{l}[lang]' for l in lang_filter])  # 组合成 OR 表达式
        lang_clause = f' AND ({langs})'  # 拼接到检索式
    return f'({base_query}) AND {review_clause} AND {date_clause}{lang_clause}'  # 返回完整检索式

# -------------------- 3) 分页拉 PMID --------------------
def paginate_pmids(query: str, quota: int, page: int = 250) -> List[str]:  # 分页获取PMID
    """
    用 retstart 分页抓 PMID，直到达到 quota 或无更多结果。  # 函数说明
    """
    pmids: List[str] = []  # 存放结果PMID列表
    retstart = 0  # 初始化分页起点
    while len(pmids) < quota:  # 未达到配额则继续
        chunk = fetch.pmids_for_query(query, retmax=min(page, quota - len(pmids)), retstart=retstart)  # 拉一页PMID
        if not chunk:  # 若无结果
            break  # 终止循环
        pmids.extend(chunk)  # 追加到结果列表
        if len(chunk) < min(page, quota - len(pmids)):  # 若返回不足一页，说明没更多了
            break  # 跳出
        retstart += len(chunk)  # 移动分页起点
    return pmids  # 返回PMID列表

# -------------------- 4) 打分组件（新近性 + 影响力 + 可选相关性） --------------------
def recency_norm(pubdate: str, year_min: int, year_max: int) -> float:  # 计算新近性归一化分
    try:
        y = int(str(pubdate)[:4])  # 从pubdate截取年份
    except Exception:
        return 0.5  # 缺失年份给中性值
    y = max(min(y, year_max), year_min)  # 限制在范围内
    return (y - year_min) / max(1, (year_max - year_min))  # 线性映射到[0,1]

def impact_norm_by_year(pmid: str, year: int,  # 估算影响力：同年内做zscore
                        cache: Dict[str, int],  # 缓存每篇被引近似值
                        year_stats: Dict[int, List[int]]) -> float:  # 记录每年被引计数分布
    """
    用 related_pmids(pmid).get('citedin', []) 的数量近似“影响力”，  # 方法说明
    统计值放到对应年份的分布，用于后续 zscore。  # 供zscore标准化
    """
    if pmid not in cache:  # 若缓存没有
        try:
            rel = fetch.related_pmids(pmid) or {}  # 获取相关文献字典
            cited = rel.get('citedin', []) or []  # 取citedin列表，近似被引/相似强度
            cache[pmid] = len(cited)  # 记录被引数量
        except Exception:
            cache[pmid] = 0  # 异常则记0
    year_stats.setdefault(year, []).append(cache[pmid])  # 将该数值加入对应年份分布
    return float(cache[pmid])  # 返回原始影响力值

def zscore(x: float, arr: List[float]) -> float:  # 计算z分数
    if not arr:  # 若分布为空
        return 0.0  # 返回0
    mu = sum(arr) / len(arr)  # 均值
    var = sum((a - mu) ** 2 for a in arr) / max(1, len(arr) - 1)  # 方差（无偏估计）
    std = math.sqrt(var) if var > 0 else 1.0  # 标准差，避免除0
    return (x - mu) / std  # 返回z分数

def score_article(article,  # 计算文章综合分
                  cited_cache: Dict[str, int],  # 被引缓存
                  year_stats: Dict[int, List[int]],  # 每年被引分布
                  y_min: int, y_max: int,  # 年份范围
                  alpha: float = 0.6,  # 新近性权重
                  beta: float = 0.4) -> float:  # 影响力权重
    """
    综合分 = 新近性(α) + 影响力(β)  # 评分公式说明
    """
    try:
        y = int(str(article.pubdate)[:4])  # 提取年份
    except Exception:
        y = y_min  # 缺失年份按最小年处理
    r = recency_norm(str(article.pubdate), y_min, y_max)  # 计算新近性分
    imp_raw = impact_norm_by_year(article.pmid, y, cited_cache, year_stats)  # 获取原始影响力值
    imp_z = zscore(imp_raw, year_stats.get(y, []))  # 转为按年z分
    return alpha * r + beta * (0.5 + 0.5 * math.tanh(imp_z))  # 返回综合分（对z分用tanh平滑）

# -------------------- 5) 年份配额与多样化 --------------------
def allocate_quota(years: List[int], batch_size: int,  # 为每年分配候选配额
                   lambda_decay: float = 0.35, min_floor: int = 6) -> Dict[int, int]:
    """
    指数衰减配额 + 地板配额（近期年权重大）。  # 分配策略说明
    """
    weights = {y: math.exp(-lambda_decay * (max(years) - y)) for y in years}  # 计算每年的权重（越近越大）
    s = sum(weights.values()) or 1.0  # 权重和，避免0
    alloc = {y: max(min_floor, int(batch_size * (weights[y] / s))) for y in years}  # 初始按权重分配并加地板
    total = sum(alloc.values())  # 计算总配额
    if total > batch_size:  # 若超出batch_size
        ratio = batch_size / total  # 计算缩放比例
        for y in sorted(years, reverse=True):  # 从近到远调整
            if sum(alloc.values()) <= batch_size:  # 达标则停止
                break  # 结束循环
            if alloc[y] > min_floor:  # 仅缩减高于地板的年份
                delta = max(1, int((alloc[y] - min_floor) * (1 - ratio)))  # 计算缩减量
                alloc[y] = max(min_floor, alloc[y] - delta)  # 应用缩减且不低于地板
    return alloc  # 返回每年的配额字典

def diversified_topk(candidates: List[Tuple[float, int, object]],  # 候选为(分数, 年份, 文章对象)
                     K: int, year_max_ratio: float = 0.4):  # 选TopK并限制单年占比
    """
    防止同一年“刷屏”。cap=⌊K*year_max_ratio⌋，不足再无条件补齐。  # 函数说明
    candidates: [(score, year, article)] 已按分数降序  # 参数说明
    """
    by_year = defaultdict(int)  # 记录每年已选数量
    out = []  # 最终输出列表
    cap = max(1, int(K * year_max_ratio))  # 单年最大占比上限（至少1）
    for s, y, a in candidates:  # 遍历降序候选
        if len(out) >= K:  # 如果已满K
            break  # 结束
        if by_year[y] >= cap:  # 若该年已达上限
            continue  # 跳过该项
        out.append((s, y, a))  # 接受该候选
        by_year[y] += 1  # 计数+1
    if len(out) < K:  # 若不足K
        seen = {id(a) for _, _, a in out}  # 已选文章的id集合
        for s, y, a in candidates:  # 再次遍历
            if len(out) >= K:  # 填满则停
                break  # 跳出
            if id(a) in seen:  # 跳过已选
                continue  # 继续
            out.append((s, y, a))  # 无条件补齐
    return out[:K]  # 返回TopK候选

# -------------------- 6) 核心：仅检索综述的批次搜索 --------------------
def batch_search_reviews(base_query: str,  # 基础主题检索式（不含日期）
                         years_back: int = 10,  # 回溯年数（近N年）
                         batch_size: int = 60,  # 每批候选总配额
                         topk_batch: int = 10,  # 每批输出TopK
                         K_total: int = 30,  # 总共希望得到的条数
                         strong_review: bool = False,  # 是否使用强综述类型
                         lang_filter: Optional[List[str]] = None,  # 语言过滤
                         lambda_decay: float = 0.35,  # 年份权重指数衰减参数
                         year_max_ratio: float = 0.4):  # 单年占比上限
    
    start_time = time.time()
    logger.info("🚀 Starting batch search for reviews...")
    logger.info(f"📊 Parameters: years_back={years_back}, batch_size={batch_size}, K_total={K_total}")
    """
    返回结构化列表（不取全文）：  # 返回结果说明
      [
        {'pmid': str, 'title': str, 'pubdate': str, 'journal': str,
         'score': float, 'year': int, 'mesh': List[str] | None},
        ...
      ]
    """
    this_year = datetime.now().year  # 当前年份
    years = list(range(this_year - years_back + 1, this_year + 1))  # 构造年份列表（含今年）

    selected: List[Dict] = []  # 已选结果列表（结构化字典）
    selected_pmids: set = set()  # 已选PMID集合（去重用）
    seen_pmids: set = set()  # 已见过的PMID集合（跨批去重）
    cited_cache: Dict[str, int] = {}  # 影响力缓存：pmid->被引近似数
    year_stats: Dict[int, List[int]] = {}  # 每年被引分布用于zscore

    batch_count = 0
    while len(selected) < K_total:  # 若未达到总目标
        batch_count += 1
        batch_start_time = time.time()
        logger.info(f"\n🔄 Starting batch {batch_count} (current progress: {len(selected)}/{K_total})")

        alloc = allocate_quota(years, batch_size=batch_size,  # 分配本批各年份配额
                               lambda_decay=lambda_decay, min_floor=6)  # 使用指数衰减+地板
        logger.info(f"📊 Year quota allocation: {dict(alloc)}")

        pool: List[Tuple[float, int, object]] = []  # 本批候选池（分数, 年份, 文章）
        with tqdm(total=len(years), desc=f"Processing years", unit="year") as pbar:
            for y in years:  # 遍历每个年份
                query = build_query(base_query, y, y,  # 构造该年的最终检索式（带Review/日期/语言）
                                    strong_review=strong_review, lang_filter=lang_filter)  # 参数传递
                quota = alloc[y]  # 使用原始配额
                logger.info(f"\n📅 Processing year {y} (quota: {quota})")

                # 如果已经处理过这一年但还需要更多文章，增加配额
                if len(selected) < K_total and any(int(str(r.pubdate)[:4]) == y for r in selected):
                    extra_quota = min(quota * 2, 50)  # 最多额外增加50篇
                    logger.info(f"  ℹ️ Adding extra quota (+{extra_quota}) to find more articles")
                    quota += extra_quota

                pmids = paginate_pmids(query, quota=quota, page=250)  # 分页抓PMID
                if pmids:
                    logger.info(f"  📑 Found {len(pmids)} articles for year {y}")
                    with tqdm(total=len(pmids), desc=f"Processing articles", unit="article") as article_pbar:
                        for pmid in pmids:  # 遍历该年PMID
                            if pmid in seen_pmids:  # 若已处理
                                continue  # 跳过
                            try:
                                a = fetch.article_by_pmid(pmid)  # 获取文章元数据
                                if not a:  # 若为空
                                    continue  # 跳过
                                
                                yy = int(str(a.pubdate)[:4])  # 提取年份
                                s = score_article(a, cited_cache, year_stats,  # 计算综合分
                                              min(years), max(years))
                                pool.append((s, yy, a))  # 加入候选池
                                seen_pmids.add(pmid)  # 标记已见
                                
                                    
                                article_pbar.update(1)  # 更新文章进度条
                                
                            except Exception as e:
                                logger.debug(f"  ⚠️ Failed to process article {pmid}: {str(e)}")
                                continue  # 异常忽略
                                
                pbar.update(1)  # 更新年份进度条

        if not pool:  # 若本批无候选
            logger.info("⚠️ No candidates found in this batch")
            if len(selected) < K_total:
                logger.info(f"🔄 Resetting seen_pmids to try finding more articles (have {len(selected)}, need {K_total})")
                seen_pmids.clear()  # 清除已见标记，允许重新处理之前的文章
                continue  # 继续下一批次
            break  # 结束循环

        logger.info(f"\n🔄 Processing batch results (found {len(pool)} candidates)")
        pool.sort(key=lambda x: x[0], reverse=True)  # 按分数降序排序候选池
        # 计算这一批需要选择多少文章
        remaining = K_total - len(selected)
        current_batch_size = min(remaining * 2, topk_batch * 2)  # 选择更多候选，但不超过两倍的topk_batch
        batch_pick = diversified_topk(pool, current_batch_size, year_max_ratio=year_max_ratio)  # 应用多样化取TopK
        logger.info(f"✅ Selected {len(batch_pick)} articles after diversity filtering (aiming for {remaining} more)")

        added_count = 0
        for score, yy, art in batch_pick:  # 遍历本批选出的文章
            if len(selected) >= K_total:  # 如果已满足总量
                break  # 停止添加
            if art.pmid in selected_pmids:  # 若该篇已在最终结果
                continue  # 跳过
            # 将分数添加到article对象
            setattr(art, 'score', float(score))
            selected.append(art)  # 直接添加article对象
            selected_pmids.add(art.pmid)  # 记录已选PMID
            added_count += 1

        batch_time = time.time() - batch_start_time
        logger.info(f"✅ Batch {batch_count} completed in {batch_time:.2f}s (added {added_count} articles)")
        logger.info(f"📊 Current progress: {len(selected)}/{K_total} articles")

        # （可选）加温：下一批略微增大新近偏置（此处保留接口）  # 可在此调整lambda_decay
        # lambda_decay *= 1.05  # 若希望下一批更偏向近期，可解开

    def _final_key(article):  # 定义最终排序键函数
        try:
            return (-article.score, -int(str(article.pubdate)[:4]), int(article.pmid))  # 先按分数降序，再按年份降序，再按PMID升序
        except Exception:
            return (-getattr(article, 'score', 0), 0, 10**12)  # 兜底键

    logger.info("\n🔄 Performing final sorting and cleanup...")
    selected.sort(key=_final_key)  # 对最终结果排序
    
    total_time = time.time() - start_time
    logger.info(f"\n✨ Search completed in {total_time:.2f}s")
    logger.info(f"📊 Final statistics:")
    logger.info(f"  - Total articles found: {len(selected)}")
    years = [int(str(a.pubdate)[:4]) for a in selected]
    logger.info(f"  - Years covered: {min(years)} - {max(years)}")
    logger.info(f"  - Average score: {sum(getattr(a, 'score', 0) for a in selected)/len(selected):.3f}")
    
    return selected[:K_total]  # 返回前K_total条

# -------------------- 7) 从自然语言到批次检索的一键封装 --------------------
def batch_search_reviews_from_user_query(  # 入口函数：自然语言→检索结果
    user_query: str,  # 自然语言问题
    years_back: int = 10,  # 近N年作为搜索池
    batch_size: int = 60,  # 每批候选配额
    topk_batch: int = 10,  # 每批产出数量
    K_total: int = 30,  # 总产出数量
    strong_review: bool = False,  # 是否包含系统综述/Meta分析
    lang_filter: Optional[List[str]] = None,  # 语言过滤
    lambda_decay: float = 0.35,  # 年份指数衰减参数
    year_max_ratio: float = 0.4  # 单年占比上限
):
    logger.info("\n" + "="*80)
    logger.info("🚀 Starting PubMed Review Search")
    logger.info("="*80)
    logger.info(f"📝 Query: {user_query}")
    logger.info(f"📊 Search parameters:")
    logger.info(f"  - Years back: {years_back}")
    logger.info(f"  - Target articles: {K_total}")
    logger.info(f"  - Strong review only: {strong_review}")
    logger.info(f"  - Language filter: {lang_filter}")
    
    start_time = time.time()
    logger.info("\n🤖 Step 1/2: Generating PubMed query...")
    base_query = llm_query_from_user_question(user_query)  # 调LLM生成基础检索式
    if "review[publication type]" not in base_query.lower():  # 若LLM漏了Review限定
        base_query = f"({base_query}) AND Review[Publication Type]"  # 自动补上综述限定
        logger.info("✅ Added Review[Publication Type] filter")
    
    logger.info("\n🔍 Step 2/2: Executing batch search...")
    results = batch_search_reviews(  # 调用核心检索函数
        base_query=base_query,  # 传入基础检索式
        years_back=years_back,  # 年份范围
        batch_size=batch_size,  # 批次配额
        topk_batch=topk_batch,  # 批内TopK
        K_total=K_total,  # 总量
        strong_review=strong_review,  # 强综述开关
        lang_filter=lang_filter,  # 语言过滤
        lambda_decay=lambda_decay,  # 衰减参数
        year_max_ratio=year_max_ratio  # 单年占比
    )  # 返回结构化结果列表

    total_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info("✨ Search Complete!")
    logger.info("="*80)
    logger.info(f"📊 Found {len(results)}/{K_total} requested articles in {total_time:.2f}s")
    if results:
        years = [int(str(r.pubdate)[:4]) for r in results]
        logger.info(f"📅 Year range: {min(years)} - {max(years)}")
        logger.info(f"📈 Score range: {min(getattr(r, 'score', 0) for r in results):.3f} - {max(getattr(r, 'score', 0) for r in results):.3f}")
        if len(results) < K_total:
            logger.warning(f"⚠️ Note: Only found {len(results)} articles, less than requested {K_total}")
    logger.info("="*80)
    
    return results

# -------------------- 8) 示例 --------------------
if __name__ == "__main__":  # 仅脚本直接运行时执行
    print("\n" + "="*80)
    print("🚀 PubMed Review Search Demo")
    print("="*80 + "\n")

    user_query = "Causal mechanisms linking diabetes and cardiovascular disease and potential therapeutic targets"  # 示例自然语言问题
    print(f"🔍 Query: {user_query}\n")

    results = batch_search_reviews_from_user_query(  # 调用一键检索
        user_query=user_query,  # 传入问题
        years_back=5,           # 近5年
        batch_size=60,          # 每批候选
        topk_batch=10,          # 每批取10
        K_total=30,             # 总共要30
        strong_review=False,    # 是否使用强综述：此处否
        lang_filter=["english"],# 只要英文文献（可去掉此参数）
        lambda_decay=0.35,      # 越大越偏近期
        year_max_ratio=0.4,     # 单年最多占40%
    )

    print("\n" + "="*80)
    print("📊 SEARCH RESULTS")
    print("="*80)
    print(f"\nFound {len(results)} articles in total")
    
    if results:
        print(f"\n🎯 ALL {len(results)} RESULTS:")
        print("-" * 80)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r.title}")
            print(f"   PMID: {r.pmid} | Year: {int(str(r.pubdate)[:4])} | Score: {getattr(r, 'score', 0):.3f}")
            print(f"   Journal: {r.journal}")
            if i % 10 == 0 and i < len(results):  # 每10条结果添加一个分隔线
                print("\n" + "-" * 40 + f" Result {i}/{len(results)} " + "-" * 40)
    else:
        print("\n❌ No results found")

    print("\n" + "="*80)
    print("✨ Demo completed!")
    print("="*80)
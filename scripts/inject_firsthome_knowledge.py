"""
注入首套刚需专题知识到 ChromaDB
运行: python scripts/inject_firsthome_knowledge.py
"""

import sys
sys.path.insert(0, ".")

from src.knowledge_base.embedder import Embedder
from src.knowledge_base.vector_store import KnowledgeIndex
from src.config import config

KNOWLEDGE_ENTRIES = [
    {
        "text": "首套刚需购房者应优先关注通勤效率。从张江、漕河泾、陆家嘴、静安寺四大核心就业区出发，45分钟地铁通勤圈内的板块是最优选择。张江辐射唐镇(2号线25分钟)、周浦(16号线)、康桥；漕河泾辐射九亭(9号线20分钟)、泗泾、莘庄；陆家嘴辐射联洋、北蔡、金桥；静安寺辐射江桥、丰庄。通勤时间每增加15分钟，房价通常降低15-25%。",
        "metadata": {"source": "firsthome_knowledge", "logic_tags": "首套刚需,通勤分析", "areas": "唐镇,周浦,康桥,九亭,泗泾,莘庄,联洋,北蔡,金桥,江桥,丰庄"},
    },
    {
        "text": "上海首套房总价门槛分档：200万以下仅能买崇明、金山远郊老公房；200-300万可买外环外小两房(泗泾、九亭、江桥)；300-400万可选中外环间紧凑两房(周浦、上大、顾村)；400-600万可买中环附近两房(北蔡、大华、金桥)；600-800万可买内中环两房或三房(大宁、长风、古北)。建议月供不超过家庭月收入的40%，首付最低35%(首套)。",
        "metadata": {"source": "firsthome_knowledge", "logic_tags": "首套刚需,总价门槛,首付月供", "areas": "崇明,金山,泗泾,九亭,江桥,周浦,上大,顾村,北蔡,大华,金桥,大宁,长风,古北"},
    },
    {
        "text": "首套刚需选房逻辑：第一看通勤，第二看总价，第三看增值潜力，第四看流动性。户型优先选择南北通透的2室或小3室(89-100㎡)，面积太小(50㎡以下)的房源未来置换困难。房龄建议在15年以内，超过20年的老房子贷款难且增值慢。电梯房比同面积楼梯房贵10-15%，但流动性好得多。",
        "metadata": {"source": "firsthome_knowledge", "logic_tags": "首套刚需,选房逻辑", "areas": ""},
    },
    {
        "text": "300万预算在上海买首套房的策略：最推荐泗泾和江桥。泗泾9号线到漕河泾25分钟、到徐家汇35分钟，300万能买2010年后电梯小两房(65-75㎡)。江桥13号线到静安寺30分钟、到大虹桥15分钟，300万可买2015年后动迁小两房。这两个板块的共同优点是总价门槛低、通勤便利、有轨道交通支撑、周边配套逐步完善。",
        "metadata": {"source": "firsthome_knowledge", "logic_tags": "首套刚需,300万预算", "areas": "泗泾,江桥"},
    },
    {
        "text": "500万预算首套购房推荐板块：周浦(18号线到龙阳路20分钟，500万能买2015年后次新两房85㎡)、上大(7号线到静安寺30分钟，500万可买次新两房80㎡)、颛桥(5号线到莘庄10分钟，500万可买次新两房90㎡)。这三个板块的共同特点是：轨道交通已兑现、有商业配套规划、板块均价在市均价以下。",
        "metadata": {"source": "firsthome_knowledge", "logic_tags": "首套刚需,500万预算", "areas": "周浦,上大,颛桥"},
    },
    {
        "text": "首套刚需：新房vs二手次新的选择逻辑。新房优势：税费低(仅契税)、房龄新、贷款容易、有可能摇到限价盘。劣势：交付周期长(2-3年)、周边配套可能不成熟、期房风险。二手次新优势：即买即住、配套可见、面积实在(得房率高)、可砍价。劣势：税费高(个税+增值税+契税可能有3-8%)、房龄已有折旧。建议：不急着住选新房，急着住或有小孩上学选二手次新。",
        "metadata": {"source": "firsthome_knowledge", "logic_tags": "首套刚需,新房vs二手", "areas": ""},
    },
    {
        "text": "首套刚需公积金贷款策略：上海家庭公积金贷款上限120万(有补充公积金)，首套利率2.85%。建议尽可能贷满公积金，不足部分用商业贷款补充(组合贷)。月供测算公式：贷款100万/30年/利率3.15% ≈ 月供4300元。首套契税优惠：90㎡以下1%，90㎡以上1.5%。",
        "metadata": {"source": "firsthome_knowledge", "logic_tags": "首套刚需,公积金,贷款策略", "areas": ""},
    },
    {
        "text": "2025-2026年首套刚需上车时机判断：当前上海二手房价格较高点回落15-25%，首套利率处于历史低位(3.15%)，政策面支持刚需(首付比例降低、利率下行)。对于自住需求的刚需来说，当前是较好的上车窗口。但需注意：不要买在价格还在下跌的趋势中，建议关注连续3个月成交量回暖的板块再出手。",
        "metadata": {"source": "firsthome_knowledge", "logic_tags": "首套刚需,时机判断", "areas": ""},
    },
    {
        "text": "首套刚需选板块要避开三类陷阱：一是过度炒作的网红板块(倒挂盘交付后可能大量抛售)；二是配套未兑现的远郊板块(规划延期风险大)；三是房龄超过25年的市中心老破小(流动性差、贷款难、增值慢)。建议选择轨道交通已通车、商业配套已开业、板块有真实居住需求的区域。",
        "metadata": {"source": "firsthome_knowledge", "logic_tags": "首套刚需,避坑指南", "areas": ""},
    },
    {
        "text": "首套房购买力评估四步法：第一步算首付(家庭储蓄+父母支持+可借款)；第二步算月供能力(家庭月收入×40%-现有负债)；第三步反推可贷款总额；第四步得出可承受总价。举例：家庭年收入50万、首付150万、无负债→可承受总价约450-500万(贷款300万/30年/月供1.3万/占收入31%)。",
        "metadata": {"source": "firsthome_knowledge", "logic_tags": "首套刚需,购买力评估", "areas": ""},
    },
]


def main():
    embedder = Embedder()
    kb = KnowledgeIndex()
    texts = [e["text"] for e in KNOWLEDGE_ENTRIES]
    metadatas = [e["metadata"] for e in KNOWLEDGE_ENTRIES]
    ids = [f"fh_{hash(e['text']) % 10**8:08x}" for e in KNOWLEDGE_ENTRIES]
    embeddings = embedder.embed(texts)
    kb.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print(f"✅ 首套刚需知识注入完成: {len(KNOWLEDGE_ENTRIES)} 条, 集合={config.knowledge_collection}")


if __name__ == "__main__":
    main()

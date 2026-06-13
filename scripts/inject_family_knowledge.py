"""
注入家庭改善专题知识到 ChromaDB
运行: python scripts/inject_family_knowledge.py
"""

import sys
sys.path.insert(0, ".")

from src.knowledge_base.embedder import Embedder
from src.knowledge_base.vector_store import KnowledgeIndex
from src.config import config

KNOWLEDGE_ENTRIES = [
    {
        "text": "家庭改善购房的核心逻辑：卖旧买新的置换链条。第一步：评估现有房产的市场价值(不是心理价位)；第二步：确定目标房源的预算范围(卖出价+新增贷款+储蓄)；第三步：算好转手税费(满五唯一可免个税，否则1%)和买入税费。置换最怕的是两头踏空——建议先卖掉旧房锁定资金，再从容淘新房。除非市场特别热，否则不要冒险先买后卖。",
        "metadata": {"source": "family_knowledge", "logic_tags": "家庭改善,置换链条", "areas": ""},
    },
    {
        "text": "上海学区房梯队参考：一梯队学校对口小区通常比同板块非学区房贵20-40%。一梯队小学包括：建平、福山外国语、明珠、上实、静教院、一师附小、汇师、高安路一小等。注意学区对口每年可能有微调，挂牌前核实教育局最新划片。2024年后上海推行教师轮岗，一梯队学区的溢价正在缓慢收窄，但核心学区依然坚挺。",
        "metadata": {"source": "family_knowledge", "logic_tags": "家庭改善,学区分析", "areas": ""},
    },
    {
        "text": "二胎家庭户型要求：至少三房，最好是3+1或四房。夫妻主卧+两个孩子各一间+书房/客房。面积建议120-140㎡。关键考量：动静分离(卧室区与活动区分开)、双卫(早高峰不排队)、阳台尺度(要有家政空间)、收纳空间(家庭物品多)。如果预算有限，宁愿牺牲面积也要保证房间数。",
        "metadata": {"source": "family_knowledge", "logic_tags": "家庭改善,户型要求", "areas": ""},
    },
    {
        "text": "双职工家庭通勤平衡策略：夫妻双方通勤时间之和应控制在90分钟以内。建议以其中一方的工作地点为中心，另一方在轨道交通沿线找平衡点。例如一方在张江、一方在静安寺，可选择2号线沿线(中山公园、江苏路)或换乘方便的世纪大道板块。比单方极端通勤(单程>60分钟)对家庭生活质量的损害更大。",
        "metadata": {"source": "family_knowledge", "logic_tags": "家庭改善,通勤平衡", "areas": "中山公园,江苏路,世纪大道"},
    },
    {
        "text": "家庭改善税费计算：卖房税费——满五唯一免征个税；满二不唯一征1%个税；不满二征5.3%增值税+1%个税。买房税费——契税(首套90㎡以下1%/以上1.5%，二套3%)。置换一套1000万的房子，税费成本通常在20-40万之间(含中介费)。这笔钱要算在置换成本里，别等签了合同才后悔。",
        "metadata": {"source": "family_knowledge", "logic_tags": "家庭改善,税费计算", "areas": ""},
    },
    {
        "text": "改善型购房的社区儿童友好度评估：一看小区内是否有儿童游乐设施(滑梯、沙池、攀爬架)；二看周边3公里内是否有三甲医院或儿科医院；三看对口学校步行距离(最好在1公里内)；四看小区内是否有同年龄段儿童群体(可通过遛娃时间观察)。一个好的社区环境对孩子成长的影响不亚于学区本身。",
        "metadata": {"source": "family_knowledge", "logic_tags": "家庭改善,儿童友好", "areas": ""},
    },
    {
        "text": "先卖后买 vs 先买后卖：当前上海楼市成交量偏弱，适合先卖后买。先把旧房挂牌卖出(留足3-6个月成交期)，拿到资金后再慢慢选新房。这样做的好处：资金确定、议价能力强(可以付定金快速成交)、不会被迫降价卖旧房。缺点是可能会错过心仪房源，过渡期需要租房。如果看中稀缺房源，可以签长成交周期(6个月)的买卖合同。",
        "metadata": {"source": "family_knowledge", "logic_tags": "家庭改善,置换策略", "areas": ""},
    },
    {
        "text": "上海各区学区梯队概述：徐汇区整体教育最强(一梯队最密集)，但房价最高；浦东体量大，优质学区多(碧云、联洋、花木)但竞争激烈；静安教育资源均衡，一师附小+市西组合是经典选择；杨浦高校附小资源丰富(二师附小、控二)；闵行七宝教育集团化办学效果好，性价比高。选学区房建议：优先选小学+初中双优的组合，而非仅看小学。",
        "metadata": {"source": "family_knowledge", "logic_tags": "家庭改善,学区梯队", "areas": "徐汇,浦东,静安,杨浦,闵行,碧云,联洋,花木,七宝"},
    },
    {
        "text": "改善置换的时间窗口：每年3-5月是学区房成交旺季(对口划片公布前)，也是置换的最佳挂牌窗口。9-11月市场相对平淡，适合买方。建议在旺季挂牌旧房(看房人多容易卖出好价)，淡季买入新房(议价空间大)。避开春节前后(1-2月)和酷暑(7-8月)交易低谷期。",
        "metadata": {"source": "family_knowledge", "logic_tags": "家庭改善,置换时机", "areas": ""},
    },
    {
        "text": "家庭改善的电梯与停车位考量：有老人和孩子的家庭必须选电梯房，6层以下无电梯的房源直接排除。车位比至少1:0.8，最好1:1以上。新能源车普及背景下，小区是否有充电桩或预留充电条件很重要。人车分流的小区更安全，更适合有小孩的家庭。",
        "metadata": {"source": "family_knowledge", "logic_tags": "家庭改善,配套设施", "areas": ""},
    },
]


def main():
    embedder = Embedder()
    kb = KnowledgeIndex()
    texts = [e["text"] for e in KNOWLEDGE_ENTRIES]
    metadatas = [e["metadata"] for e in KNOWLEDGE_ENTRIES]
    ids = [f"fm_{hash(e['text']) % 10**8:08x}" for e in KNOWLEDGE_ENTRIES]
    embeddings = embedder.embed(texts)
    kb.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print(f"✅ 家庭改善知识注入完成: {len(KNOWLEDGE_ENTRIES)} 条")


if __name__ == "__main__":
    main()

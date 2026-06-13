"""
注入女性购房安全专题知识到 ChromaDB
运行: python scripts/inject_herhome_knowledge.py
"""

import sys
sys.path.insert(0, ".")

from src.knowledge_base.embedder import Embedder
from src.knowledge_base.vector_store import KnowledgeIndex
from src.config import config

KNOWLEDGE_ENTRIES = [
    {
        "text": "女性购房安全评估第一要素：小区夜间照明。看房时建议晚上8-10点再去一次，观察小区主干道、单元门入口、地下车库到电梯间的照明情况。照明充足、无死角的小区安全性远高于昏暗小区。重点关注：单元门到小区大门之间是否有监控覆盖，门禁是否为可视对讲系统。",
        "metadata": {"source": "herhome_knowledge", "logic_tags": "女性购房,安全评估,夜间照明", "areas": ""},
    },
    {
        "text": "女性独居购房的安保配置要求：小区应有24小时保安巡逻(而非仅门卫)，单元门禁需刷卡或人脸识别，电梯需刷卡到层。看房时注意楼道是否有杂物堆积(安全隐患)和监控摄像头是否正常工作(看有无指示灯)。首选品牌开发商物业(万科、绿城、龙湖等)管理的小区。",
        "metadata": {"source": "herhome_knowledge", "logic_tags": "女性购房,安保配置", "areas": ""},
    },
    {
        "text": '女性购房产权规划：婚前用个人财产买房，登记在自己名下，属于个人财产(需保留付款凭证)。父母出资购房，最好直接转账至开发商账户并备注"赠与本人"，而非经手现金。婚后用婚前财产置换房产，保留完整的资金流转链条证明。建议在购房前咨询专业律师做产权规划。',
        "metadata": {"source": "herhome_knowledge", "logic_tags": "女性购房,产权规划", "areas": ""},
    },
    {
        "text": "女性购房贷款政策：上海目前首套首付最低35%，二套50%(普通住宅)或70%(非普通)。银行对女性贷款人的审批独立(不看配偶负债)，但收入证明需覆盖月供2倍。自由职业女性可提供银行流水+纳税证明替代收入证明。公积金贷款上限个人60万(有补充公积金)，家庭120万。",
        "metadata": {"source": "herhome_knowledge", "logic_tags": "女性购房,贷款政策", "areas": ""},
    },
    {
        "text": "女性购房安全加分项：小区周边500米内应有便利店、药店、24小时营业场所；地铁站到小区之间应有照明和人流，避免经过地下通道或偏僻小路；小区居民自住率高(>70%)比出租率高的小区更安全；小区物业费在3元/㎡·月以上的通常安保和保洁更有保障。",
        "metadata": {"source": "herhome_knowledge", "logic_tags": "女性购房,安全加分项", "areas": ""},
    },
    {
        "text": "女性购房流动性评估：选流动性好的房子未来更好出手。流动性好的特征：总价在市场主流区间(400-800万)、户型方正(非异形)、面积适中(60-100㎡)、房龄15年内、有电梯、有地铁。不建议买超大户型(150㎡+)或极小户型(40㎡以下)，这两类接盘侠少、出手周期长。",
        "metadata": {"source": "herhome_knowledge", "logic_tags": "女性购房,流动性评估", "areas": ""},
    },
    {
        "text": "女性购房独居注意：入户动线安全——从小区大门到单元门到入户门全程是否明亮通畅；快递和外卖接收方案——有无快递柜或蜂巢箱(避免陌生人上门)；紧急呼叫路径——手机信号是否满格、是否有物业紧急联系电话；邻居构成——可通过多次不同时段看房了解实际居住人群。",
        "metadata": {"source": "herhome_knowledge", "logic_tags": "女性购房,独居安全", "areas": ""},
    },
    {
        "text": "女性购房继承与赠与：婚前父母出资购房，登记在子女名下，属于子女个人财产。婚后父母出资，登记在子女名下，若能证明是赠与一方的(有书面赠与协议)，属于个人财产，否则视为夫妻共同财产。继承房产免征个税和增值税，但需缴纳公证费(约房产价值的1-2%)。建议做遗嘱公证或生前赠与。",
        "metadata": {"source": "herhome_knowledge", "logic_tags": "女性购房,继承赠与", "areas": ""},
    },
    {
        "text": "女性友好社区特征：周边有女性常去的消费场所(瑜伽馆、美容院、咖啡馆等)；小区内白天可见老人带小孩(社区有人气、不冷清)；底商以超市、药店、花店等生活服务类为主(而非KTV、酒吧等)；距离最近的派出所/警务站不超过2公里。浦东联洋、古北、碧云是上海女性友好度较高的板块。",
        "metadata": {"source": "herhome_knowledge", "logic_tags": "女性购房,女性友好社区", "areas": "联洋,古北,碧云"},
    },
    {
        "text": "女性婚前购房建议：如果条件允许，建议婚前买一套属于自己名字的房产。优势：婚前财产独立、有安全感、婚后有底气和话语权。面积不必太大(一房或小两房即可)，地段最重要(好出租、好出手)。月供控制在个人月收入的30%以内，留足生活现金流。婚后可与配偶共同置换升级为家庭住房。",
        "metadata": {"source": "herhome_knowledge", "logic_tags": "女性购房,婚前购房", "areas": ""},
    },
]


def main():
    embedder = Embedder()
    kb = KnowledgeIndex()
    texts = [e["text"] for e in KNOWLEDGE_ENTRIES]
    metadatas = [e["metadata"] for e in KNOWLEDGE_ENTRIES]
    ids = [f"hh_{hash(e['text']) % 10**8:08x}" for e in KNOWLEDGE_ENTRIES]
    embeddings = embedder.embed(texts)
    kb.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print(f"✅ 女性购房知识注入完成: {len(KNOWLEDGE_ENTRIES)} 条")


if __name__ == "__main__":
    main()

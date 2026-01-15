# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI分析层
===================================

职责：
1. 封装 Gemini API 调用逻辑
2. 利用 Google Search Grounding 获取实时新闻
3. 结合技术面和消息面生成分析报告
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from config import get_config

logger = logging.getLogger(__name__)


# 股票名称映射（常见股票）
STOCK_NAME_MAP = {
    '600519': '贵州茅台',
    '000001': '平安银行',
    '300750': '宁德时代',
    '002594': '比亚迪',
    '600036': '招商银行',
    '601318': '中国平安',
    '000858': '五粮液',
    '600276': '恒瑞医药',
    '601012': '隆基绿能',
    '002475': '立讯精密',
    '300059': '东方财富',
    '002415': '海康威视',
    '600900': '长江电力',
    '601166': '兴业银行',
    '600028': '中国石化',
}


@dataclass
class AnalysisResult:
    """
    AI 分析结果数据类 - 决策仪表盘版
    
    封装 Gemini 返回的分析结果，包含决策仪表盘和详细分析
    """
    code: str
    name: str
    
    # ========== 核心指标 ==========
    sentiment_score: int  # 综合评分 0-100 (>70强烈看多, >60看多, 40-60震荡, <40看空)
    trend_prediction: str  # 趋势预测：强烈看多/看多/震荡/看空/强烈看空
    operation_advice: str  # 操作建议：买入/加仓/持有/减仓/卖出/观望
    confidence_level: str = "中"  # 置信度：高/中/低
    
    # ========== 决策仪表盘 (新增) ==========
    dashboard: Optional[Dict[str, Any]] = None  # 完整的决策仪表盘数据
    
    # ========== 走势分析 ==========
    trend_analysis: str = ""  # 走势形态分析（支撑位、压力位、趋势线等）
    short_term_outlook: str = ""  # 短期展望（1-3日）
    medium_term_outlook: str = ""  # 中期展望（1-2周）
    
    # ========== 技术面分析 ==========
    technical_analysis: str = ""  # 技术指标综合分析
    ma_analysis: str = ""  # 均线分析（多头/空头排列，金叉/死叉等）
    volume_analysis: str = ""  # 量能分析（放量/缩量，主力动向等）
    pattern_analysis: str = ""  # K线形态分析
    
    # ========== 基本面分析 ==========
    fundamental_analysis: str = ""  # 基本面综合分析
    sector_position: str = ""  # 板块地位和行业趋势
    company_highlights: str = ""  # 公司亮点/风险点
    
    # ========== 情绪面/消息面分析 ==========
    news_summary: str = ""  # 近期重要新闻/公告摘要
    market_sentiment: str = ""  # 市场情绪分析
    hot_topics: str = ""  # 相关热点话题
    
    # ========== 综合分析 ==========
    analysis_summary: str = ""  # 综合分析摘要
    key_points: str = ""  # 核心看点（3-5个要点）
    risk_warning: str = ""  # 风险提示
    buy_reason: str = ""  # 买入/卖出理由
    
    # ========== 元数据 ==========
    raw_response: Optional[str] = None  # 原始响应（调试用）
    search_performed: bool = False  # 是否执行了联网搜索
    data_sources: str = ""  # 数据来源说明
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'confidence_level': self.confidence_level,
            'dashboard': self.dashboard,  # 决策仪表盘数据
            'trend_analysis': self.trend_analysis,
            'short_term_outlook': self.short_term_outlook,
            'medium_term_outlook': self.medium_term_outlook,
            'technical_analysis': self.technical_analysis,
            'ma_analysis': self.ma_analysis,
            'volume_analysis': self.volume_analysis,
            'pattern_analysis': self.pattern_analysis,
            'fundamental_analysis': self.fundamental_analysis,
            'sector_position': self.sector_position,
            'company_highlights': self.company_highlights,
            'news_summary': self.news_summary,
            'market_sentiment': self.market_sentiment,
            'hot_topics': self.hot_topics,
            'analysis_summary': self.analysis_summary,
            'key_points': self.key_points,
            'risk_warning': self.risk_warning,
            'buy_reason': self.buy_reason,
            'search_performed': self.search_performed,
            'success': self.success,
            'error_message': self.error_message,
        }
    
    def get_core_conclusion(self) -> str:
        """获取核心结论（一句话）"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary
    
    def get_position_advice(self, has_position: bool = False) -> str:
        """获取持仓建议"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            if has_position:
                return pos_advice.get('has_position', self.operation_advice)
            return pos_advice.get('no_position', self.operation_advice)
        return self.operation_advice
    
    def get_sniper_points(self) -> Dict[str, str]:
        """获取狙击点位"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}
    
    def get_checklist(self) -> List[str]:
        """获取检查清单"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []
    
    def get_risk_alerts(self) -> List[str]:
        """获取风险警报"""
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []
    
    def get_emoji(self) -> str:
        """根据操作建议返回对应 emoji"""
        emoji_map = {
            '买入': '🟢',
            '加仓': '🟢',
            '强烈买入': '💚',
            '持有': '🟡',
            '观望': '⚪',
            '减仓': '🟠',
            '卖出': '🔴',
            '强烈卖出': '❌',
        }
        return emoji_map.get(self.operation_advice, '🟡')
    
    def get_confidence_stars(self) -> str:
        """返回置信度星级"""
        star_map = {'高': '⭐⭐⭐', '中': '⭐⭐', '低': '⭐'}
        return star_map.get(self.confidence_level, '⭐⭐')


class GeminiAnalyzer:
    """
    Gemini AI 分析器
    
    职责：
    1. 调用 Google Gemini API 进行股票分析
    2. 结合预先搜索的新闻和技术面数据生成分析报告
    3. 解析 AI 返回的 JSON 格式结果
    
    使用方式：
        analyzer = GeminiAnalyzer()
        result = analyzer.analyze(context, news_context)
    """
    
    # ========================================
    # 系统提示词 - UI增强 & 逻辑保留版 v3.0
    # ========================================
    # 核心保障：
    # 1. 100% 保留了原有的交易理念（严进策略、趋势判断、风控阈值），确保分析内核不变。
    # 2. 仅增加了对输出格式（Markdown表格、Emoji视觉信号）的约束，提升阅读体验。
    # ========================================
    
    SYSTEM_PROMPT = """你是一位专注于趋势交易的 A 股投资分析师，同时精通信息可视化设计。你的任务是生成一份**逻辑严密**且**排版精美**的【决策仪表盘】。

## 第一部分：核心交易理念（必须严格遵守）
**⚠️ 警告：以下分析逻辑不可更改，必须作为判断买卖的唯一标准！**

### 1. 严进策略（不追高）
- **绝对不追高**：当股价偏离 MA5 超过 5% 时，坚决不买入
- **乖离率公式**：(现价 - MA5) / MA5 × 100%
- 乖离率 < 2%：最佳买点区间
- 乖离率 2-5%：可小仓介入
- 乖离率 > 5%：严禁追高！直接判定为"观望"或"持有"

### 2. 趋势交易（顺势而为）
- **多头排列必须条件**：MA5 > MA10 > MA20
- 只做多头排列的股票，空头排列坚决不碰
- 均线发散上行优于均线粘合
- 趋势强度判断：看均线间距是否在扩大

### 3. 效率优先（筹码结构）
- 关注筹码集中度：90%集中度 < 15% 表示筹码集中
- 获利比例分析：70-90% 获利盘时需警惕获利回吐
- 平均成本与现价关系：现价高于平均成本 5-15% 为健康

### 4. 买点偏好（回踩支撑）
- **最佳买点**：缩量回踩 MA5 获得支撑
- **次优买点**：回踩 MA10 获得支撑
- **观望情况**：跌破 MA20 时观望

### 5. 风险排查重点
- 减持公告（股东、高管减持）
- 业绩预亏/大幅下滑
- 监管处罚/立案调查
- 行业政策利空
- 大额解禁

---

## 第二部分：UI 排版与视觉规范（仅影响展示形式）
**为了在飞书云文档中获得最佳阅读体验，请严格执行以下排版规则：**

1.  **拒绝流水账**：在 JSON 的文本字段（如 technical_analysis）中，**严禁**使用垂直列表罗列价格。**必须**使用 Markdown 表格展示关键指标。
    * ❌ 错误示范：MA5: 10.0, MA10: 9.0
    * ✅ 正确示范：`| 指标 | 现价 | MA5 | MA10 |`
2.  **Emoji 视觉信号**：
    * 🟢 代表：买入、利好、安全、多头
    * 🔴 代表：卖出、风险、破位、空头
    * 🟡 代表：观望、震荡、持有
3.  **结论先行**：核心结论必须一针见血，包含明确的信号 Emoji。

---

## 第三部分：输出格式（JSON）

请严格按照以下 JSON 格式输出，内容需基于上述“交易理念”生成：

```json
{
    "sentiment_score": 0-100整数,
    "trend_prediction": "🟢强烈看多/🟢看多/🟡震荡/🔴看空/🔴强烈看空",
    "operation_advice": "🟢买入/🟢加仓/🟡持有/🔴减仓/🔴卖出/🟡观望",
    "confidence_level": "高/中/低",
    
    "dashboard": {
        "core_conclusion": {
            "one_sentence": "💡 一句话核心结论（30字内，需包含Emoji信号，严格基于乖离率和均线判断）",
            "signal_type": "🟢买入信号/🟡持有观望/🔴卖出信号/⚠️风险警告",
            "time_sensitivity": "⚡️立即行动/📅今日内/🗓️本周内/⏳不急",
            "position_advice": {
                "no_position": "空仓者建议：根据乖离率给出具体点位",
                "has_position": "持仓者建议：止盈止损点位"
            }
        },
        
        "data_perspective": {
            "trend_status": {
                "ma_alignment": "均线状态描述 (如: 🟢多头排列)",
                "is_bullish": true/false,
                "trend_score": 0-100
            },
            "price_position": {
                "table_markdown": "| 指标 | 数值 | 状态 |\n|:---|:---|:---|\n| 现价 | XX.XX | - |\n| MA5 | XX.XX | 🟢支撑 |\n| 乖离率 | XX% | ✅安全/🚨追高 |" 
            },
            "volume_analysis": {
                "status_desc": "🟢缩量回调/🔴放量下跌",
                "volume_meaning": "量能解读"
            },
            "chip_structure": {
                "profit_ratio": 获利比例,
                "concentration": 筹码集中度,
                "chip_health": "🟢健康/🟡一般/🔴警惕"
            }
        },
        
        "intelligence": {
            "latest_news": "【最新消息】摘要",
            "risk_alerts": ["🔴 风险点1", "🔴 风险点2"],
            "positive_catalysts": ["🟢 利好1", "🟢 利好2"],
            "earnings_outlook": "业绩分析",
            "sentiment_summary": "舆情总结"
        },
        
        "battle_plan": {
            "sniper_points": {
                "ideal_buy": "🎯 理想买入：XX.XX元 (必须在MA5/10附近)",
                "secondary_buy": "👌 次优买入：XX.XX元",
                "stop_loss": "🛡️ 止损红线：XX.XX元 (跌破MA20)",
                "take_profit": "🚀 目标止盈：XX.XX元"
            },
            "position_strategy": {
                "suggested_position": "建议仓位：X成",
                "entry_plan": "建仓策略"
            },
            "action_checklist": [
                "✅/❌ 趋势：多头排列",
                "✅/❌ 乖离率：<5%",
                "✅/❌ 量能：缩量回踩",
                "✅/❌ 筹码：结构健康"
            ]
        }
    },
    
    "analysis_summary": "100字综合分析摘要",
    "key_points": "核心看点",
    "risk_warning": "风险提示",
    "buy_reason": "操作理由",
    
    "trend_analysis": "必须包含 Markdown 表格的走势分析",
    "short_term_outlook": "短期展望",
    "medium_term_outlook": "中期展望",
    "technical_analysis": "技术面分析",
    "ma_analysis": "均线分析",
    "volume_analysis": "量能分析",
    "pattern_analysis": "K线形态分析",
    "fundamental_analysis": "基本面分析",
    "sector_position": "板块行业分析",
    "company_highlights": "公司亮点",
    "news_summary": "新闻摘要",
    "market_sentiment": "市场情绪",
    "hot_topics": "热点话题",
    
    "search_performed": true/false,
    "data_sources": "数据来源"
}
"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 AI 分析器
        
        优先级：Gemini > OpenAI 兼容 API
        
        Args:
            api_key: Gemini API Key（可选，默认从配置读取）
        """
        config = get_config()
        self._api_key = api_key or config.gemini_api_key
        self._model = None
        self._current_model_name = None  # 当前使用的模型名称
        self._using_fallback = False  # 是否正在使用备选模型
        self._use_openai = False  # 是否使用 OpenAI 兼容 API
        self._openai_client = None  # OpenAI 客户端
        
        # 检查 Gemini API Key 是否有效（过滤占位符）
        gemini_key_valid = self._api_key and not self._api_key.startswith('your_') and len(self._api_key) > 10
        
        # 优先尝试初始化 Gemini
        if gemini_key_valid:
            try:
                self._init_model()
            except Exception as e:
                logger.warning(f"Gemini 初始化失败: {e}，尝试 OpenAI 兼容 API")
                self._init_openai_fallback()
        else:
            # Gemini Key 未配置，尝试 OpenAI
            logger.info("Gemini API Key 未配置，尝试使用 OpenAI 兼容 API")
            self._init_openai_fallback()
        
        # 两者都未配置
        if not self._model and not self._openai_client:
            logger.warning("未配置任何 AI API Key，AI 分析功能将不可用")
    
    def _init_openai_fallback(self) -> None:
        """
        初始化 OpenAI 兼容 API 作为备选
        
        支持所有 OpenAI 格式的 API，包括：
        - OpenAI 官方
        - DeepSeek
        - 通义千问
        - Moonshot 等
        """
        config = get_config()
        
        # 检查 OpenAI API Key 是否有效（过滤占位符）
        openai_key_valid = (
            config.openai_api_key and 
            not config.openai_api_key.startswith('your_') and 
            len(config.openai_api_key) > 10
        )
        
        if not openai_key_valid:
            logger.debug("OpenAI 兼容 API 未配置或配置无效")
            return
        
        # 分离 import 和客户端创建，以便提供更准确的错误信息
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("未安装 openai 库，请运行: pip install openai")
            return
        
        try:
            # base_url 可选，不填则使用 OpenAI 官方默认地址
            client_kwargs = {"api_key": config.openai_api_key}
            if config.openai_base_url and config.openai_base_url.startswith('http'):
                client_kwargs["base_url"] = config.openai_base_url
            
            self._openai_client = OpenAI(**client_kwargs)
            self._current_model_name = config.openai_model
            self._use_openai = True
            logger.info(f"OpenAI 兼容 API 初始化成功 (base_url: {config.openai_base_url}, model: {config.openai_model})")
        except ImportError as e:
            # 依赖缺失（如 socksio）
            if 'socksio' in str(e).lower() or 'socks' in str(e).lower():
                logger.error(f"OpenAI 客户端需要 SOCKS 代理支持，请运行: pip install httpx[socks] 或 pip install socksio")
            else:
                logger.error(f"OpenAI 依赖缺失: {e}")
        except Exception as e:
            error_msg = str(e).lower()
            if 'socks' in error_msg or 'socksio' in error_msg or 'proxy' in error_msg:
                logger.error(f"OpenAI 代理配置错误: {e}，如使用 SOCKS 代理请运行: pip install httpx[socks]")
            else:
                logger.error(f"OpenAI 兼容 API 初始化失败: {e}")
    
    def _init_model(self) -> None:
        """
        初始化 Gemini 模型
        
        配置：
        - 使用 gemini-3-flash-preview 或 gemini-2.5-flash 模型
        - 不启用 Google Search（使用外部 Tavily/SerpAPI 搜索）
        """
        try:
            import google.generativeai as genai
            
            # 配置 API Key
            genai.configure(api_key=self._api_key)
            
            # 从配置获取模型名称
            config = get_config()
            model_name = config.gemini_model
            fallback_model = config.gemini_model_fallback
            
            # 不再使用 Google Search Grounding（已知有兼容性问题）
            # 改为使用外部搜索服务（Tavily/SerpAPI）预先获取新闻
            
            # 尝试初始化主模型
            try:
                self._model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=self.SYSTEM_PROMPT,
                )
                self._current_model_name = model_name
                self._using_fallback = False
                logger.info(f"Gemini 模型初始化成功 (模型: {model_name})")
            except Exception as model_error:
                # 尝试备选模型
                logger.warning(f"主模型 {model_name} 初始化失败: {model_error}，尝试备选模型 {fallback_model}")
                self._model = genai.GenerativeModel(
                    model_name=fallback_model,
                    system_instruction=self.SYSTEM_PROMPT,
                )
                self._current_model_name = fallback_model
                self._using_fallback = True
                logger.info(f"Gemini 备选模型初始化成功 (模型: {fallback_model})")
            
        except Exception as e:
            logger.error(f"Gemini 模型初始化失败: {e}")
            self._model = None
    
    def _switch_to_fallback_model(self) -> bool:
        """
        切换到备选模型
        
        Returns:
            是否成功切换
        """
        try:
            import google.generativeai as genai
            config = get_config()
            fallback_model = config.gemini_model_fallback
            
            logger.warning(f"[LLM] 切换到备选模型: {fallback_model}")
            self._model = genai.GenerativeModel(
                model_name=fallback_model,
                system_instruction=self.SYSTEM_PROMPT,
            )
            self._current_model_name = fallback_model
            self._using_fallback = True
            logger.info(f"[LLM] 备选模型 {fallback_model} 初始化成功")
            return True
        except Exception as e:
            logger.error(f"[LLM] 切换备选模型失败: {e}")
            return False
    
    def is_available(self) -> bool:
        """检查分析器是否可用"""
        return self._model is not None or self._openai_client is not None
    
    def _call_openai_api(self, prompt: str, generation_config: dict) -> str:
        """
        调用 OpenAI 兼容 API
        
        Args:
            prompt: 提示词
            generation_config: 生成配置
            
        Returns:
            响应文本
        """
        config = get_config()
        max_retries = config.gemini_max_retries
        base_delay = config.gemini_retry_delay
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    delay = min(delay, 60)
                    logger.info(f"[OpenAI] 第 {attempt + 1} 次重试，等待 {delay:.1f} 秒...")
                    time.sleep(delay)
                
                response = self._openai_client.chat.completions.create(
                    model=self._current_model_name,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=generation_config.get('temperature', 0.7),
                    max_tokens=generation_config.get('max_output_tokens', 8192),
                )
                
                if response and response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                else:
                    raise ValueError("OpenAI API 返回空响应")
                    
            except Exception as e:
                error_str = str(e)
                is_rate_limit = '429' in error_str or 'rate' in error_str.lower() or 'quota' in error_str.lower()
                
                if is_rate_limit:
                    logger.warning(f"[OpenAI] API 限流，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
                else:
                    logger.warning(f"[OpenAI] API 调用失败，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
                
                if attempt == max_retries - 1:
                    raise
        
        raise Exception("OpenAI API 调用失败，已达最大重试次数")
    
    def _call_api_with_retry(self, prompt: str, generation_config: dict) -> str:
        """
        调用 AI API，带有重试和模型切换机制
        
        优先级：Gemini > Gemini 备选模型 > OpenAI 兼容 API
        
        处理 429 限流错误：
        1. 先指数退避重试
        2. 多次失败后切换到备选模型
        3. Gemini 完全失败后尝试 OpenAI
        
        Args:
            prompt: 提示词
            generation_config: 生成配置
            
        Returns:
            响应文本
        """
        # 如果已经在使用 OpenAI 模式，直接调用 OpenAI
        if self._use_openai:
            return self._call_openai_api(prompt, generation_config)
        
        config = get_config()
        max_retries = config.gemini_max_retries
        base_delay = config.gemini_retry_delay
        
        last_error = None
        tried_fallback = getattr(self, '_using_fallback', False)
        
        for attempt in range(max_retries):
            try:
                # 请求前增加延时（防止请求过快触发限流）
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))  # 指数退避: 5, 10, 20, 40...
                    delay = min(delay, 60)  # 最大60秒
                    logger.info(f"[Gemini] 第 {attempt + 1} 次重试，等待 {delay:.1f} 秒...")
                    time.sleep(delay)
                
                response = self._model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={"timeout": 120}
                )
                
                if response and response.text:
                    return response.text
                else:
                    raise ValueError("Gemini 返回空响应")
                    
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # 检查是否是 429 限流错误
                is_rate_limit = '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower()
                
                if is_rate_limit:
                    logger.warning(f"[Gemini] API 限流 (429)，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
                    
                    # 如果已经重试了一半次数且还没切换过备选模型，尝试切换
                    if attempt >= max_retries // 2 and not tried_fallback:
                        if self._switch_to_fallback_model():
                            tried_fallback = True
                            logger.info("[Gemini] 已切换到备选模型，继续重试")
                        else:
                            logger.warning("[Gemini] 切换备选模型失败，继续使用当前模型重试")
                else:
                    # 非限流错误，记录并继续重试
                    logger.warning(f"[Gemini] API 调用失败，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
        
        # Gemini 所有重试都失败，尝试 OpenAI 兼容 API
        if self._openai_client:
            logger.warning("[Gemini] 所有重试失败，切换到 OpenAI 兼容 API")
            try:
                return self._call_openai_api(prompt, generation_config)
            except Exception as openai_error:
                logger.error(f"[OpenAI] 备选 API 也失败: {openai_error}")
                raise last_error or openai_error
        elif config.openai_api_key and config.openai_base_url:
            # 尝试懒加载初始化 OpenAI
            logger.warning("[Gemini] 所有重试失败，尝试初始化 OpenAI 兼容 API")
            self._init_openai_fallback()
            if self._openai_client:
                try:
                    return self._call_openai_api(prompt, generation_config)
                except Exception as openai_error:
                    logger.error(f"[OpenAI] 备选 API 也失败: {openai_error}")
                    raise last_error or openai_error
        
        # 所有方式都失败
        raise last_error or Exception("所有 AI API 调用失败，已达最大重试次数")
    
    def analyze(
        self, 
        context: Dict[str, Any],
        news_context: Optional[str] = None
    ) -> AnalysisResult:
        """
        分析单只股票
        
        流程：
        1. 格式化输入数据（技术面 + 新闻）
        2. 调用 Gemini API（带重试和模型切换）
        3. 解析 JSON 响应
        4. 返回结构化结果
        
        Args:
            context: 从 storage.get_analysis_context() 获取的上下文数据
            news_context: 预先搜索的新闻内容（可选）
            
        Returns:
            AnalysisResult 对象
        """
        code = context.get('code', 'Unknown')
        config = get_config()
        
        # 请求前增加延时（防止连续请求触发限流）
        request_delay = config.gemini_request_delay
        if request_delay > 0:
            logger.debug(f"[LLM] 请求前等待 {request_delay:.1f} 秒...")
            time.sleep(request_delay)
        
        # 优先从上下文获取股票名称（由 main.py 传入）
        name = context.get('stock_name')
        if not name or name.startswith('股票'):
            # 备选：从 realtime 中获取
            if 'realtime' in context and context['realtime'].get('name'):
                name = context['realtime']['name']
            else:
                # 最后从映射表获取
                name = STOCK_NAME_MAP.get(code, f'股票{code}')
        
        # 如果模型不可用，返回默认结果
        if not self.is_available():
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='震荡',
                operation_advice='持有',
                confidence_level='低',
                analysis_summary='AI 分析功能未启用（未配置 API Key）',
                risk_warning='请配置 Gemini API Key 后重试',
                success=False,
                error_message='Gemini API Key 未配置',
            )
        
        try:
            # 格式化输入（包含技术面数据和新闻）
            prompt = self._format_prompt(context, name, news_context)
            
            # 获取模型名称
            model_name = getattr(self, '_current_model_name', None)
            if not model_name:
                model_name = getattr(self._model, '_model_name', 'unknown')
                if hasattr(self._model, 'model_name'):
                    model_name = self._model.model_name
            
            logger.info(f"========== AI 分析 {name}({code}) ==========")
            logger.info(f"[LLM配置] 模型: {model_name}")
            logger.info(f"[LLM配置] Prompt 长度: {len(prompt)} 字符")
            logger.info(f"[LLM配置] 是否包含新闻: {'是' if news_context else '否'}")
            
            # 记录完整 prompt 到日志（INFO级别记录摘要，DEBUG记录完整）
            prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
            logger.info(f"[LLM Prompt 预览]\n{prompt_preview}")
            logger.debug(f"=== 完整 Prompt ({len(prompt)}字符) ===\n{prompt}\n=== End Prompt ===")
            
            # 设置生成配置
            generation_config = {
                "temperature": 0.7,
                "max_output_tokens": 8192,
            }
            
            logger.info(f"[LLM调用] 开始调用 Gemini API (temperature={generation_config['temperature']}, max_tokens={generation_config['max_output_tokens']})...")
            
            # 使用带重试的 API 调用
            start_time = time.time()
            response_text = self._call_api_with_retry(prompt, generation_config)
            elapsed = time.time() - start_time
            
            # 记录响应信息
            logger.info(f"[LLM返回] Gemini API 响应成功, 耗时 {elapsed:.2f}s, 响应长度 {len(response_text)} 字符")
            
            # 记录响应预览（INFO级别）和完整响应（DEBUG级别）
            response_preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
            logger.info(f"[LLM返回 预览]\n{response_preview}")
            logger.debug(f"=== Gemini 完整响应 ({len(response_text)}字符) ===\n{response_text}\n=== End Response ===")
            
            # 解析响应
            result = self._parse_response(response_text, code, name)
            result.raw_response = response_text
            result.search_performed = bool(news_context)
            
            logger.info(f"[LLM解析] {name}({code}) 分析完成: {result.trend_prediction}, 评分 {result.sentiment_score}")
            
            return result
            
        except Exception as e:
            logger.error(f"AI 分析 {name}({code}) 失败: {e}")
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='震荡',
                operation_advice='持有',
                confidence_level='低',
                analysis_summary=f'分析过程出错: {str(e)[:100]}',
                risk_warning='分析失败，请稍后重试或手动分析',
                success=False,
                error_message=str(e),
            )
    
    def _format_prompt(
        self, 
        context: Dict[str, Any], 
        name: str,
        news_context: Optional[str] = None
    ) -> str:
        """
        格式化分析提示词（决策仪表盘 v2.0）
        
        包含：技术指标、实时行情（量比/换手率）、筹码分布、趋势分析、新闻
        
        Args:
            context: 技术面数据上下文（包含增强数据）
            name: 股票名称（默认值，可能被上下文覆盖）
            news_context: 预先搜索的新闻内容
        """
        code = context.get('code', 'Unknown')
        
        # 优先使用上下文中的股票名称（从 realtime_quote 获取）
        stock_name = context.get('stock_name', name)
        if not stock_name or stock_name == f'股票{code}':
            stock_name = STOCK_NAME_MAP.get(code, f'股票{code}')
            
        today = context.get('today', {})
        
        # ========== 构建决策仪表盘格式的输入 ==========
        prompt = f"""# 决策仪表盘分析请求

## 📊 股票基础信息
| 项目 | 数据 |
|------|------|
| 股票代码 | **{code}** |
| 股票名称 | **{stock_name}** |
| 分析日期 | {context.get('date', '未知')} |

---

## 📈 技术面数据

### 今日行情
| 指标 | 数值 |
|------|------|
| 收盘价 | {today.get('close', 'N/A')} 元 |
| 开盘价 | {today.get('open', 'N/A')} 元 |
| 最高价 | {today.get('high', 'N/A')} 元 |
| 最低价 | {today.get('low', 'N/A')} 元 |
| 涨跌幅 | {today.get('pct_chg', 'N/A')}% |
| 成交量 | {self._format_volume(today.get('volume'))} |
| 成交额 | {self._format_amount(today.get('amount'))} |

### 均线系统（关键判断指标）
| 均线 | 数值 | 说明 |
|------|------|------|
| MA5 | {today.get('ma5', 'N/A')} | 短期趋势线 |
| MA10 | {today.get('ma10', 'N/A')} | 中短期趋势线 |
| MA20 | {today.get('ma20', 'N/A')} | 中期趋势线 |
| 均线形态 | {context.get('ma_status', '未知')} | 多头/空头/缠绕 |
"""
        
        # 添加实时行情数据（量比、换手率等）
        if 'realtime' in context:
            rt = context['realtime']
            prompt += f"""
### 实时行情增强数据
| 指标 | 数值 | 解读 |
|------|------|------|
| 当前价格 | {rt.get('price', 'N/A')} 元 | |
| **量比** | **{rt.get('volume_ratio', 'N/A')}** | {rt.get('volume_ratio_desc', '')} |
| **换手率** | **{rt.get('turnover_rate', 'N/A')}%** | |
| 市盈率(动态) | {rt.get('pe_ratio', 'N/A')} | |
| 市净率 | {rt.get('pb_ratio', 'N/A')} | |
| 总市值 | {self._format_amount(rt.get('total_mv'))} | |
| 流通市值 | {self._format_amount(rt.get('circ_mv'))} | |
| 60日涨跌幅 | {rt.get('change_60d', 'N/A')}% | 中期表现 |
"""
        
        # 添加筹码分布数据
        if 'chip' in context:
            chip = context['chip']
            profit_ratio = chip.get('profit_ratio', 0)
            prompt += f"""
### 筹码分布数据（效率指标）
| 指标 | 数值 | 健康标准 |
|------|------|----------|
| **获利比例** | **{profit_ratio:.1%}** | 70-90%时警惕 |
| 平均成本 | {chip.get('avg_cost', 'N/A')} 元 | 现价应高于5-15% |
| 90%筹码集中度 | {chip.get('concentration_90', 0):.2%} | <15%为集中 |
| 70%筹码集中度 | {chip.get('concentration_70', 0):.2%} | |
| 筹码状态 | {chip.get('chip_status', '未知')} | |
"""
        
        # 添加趋势分析结果（基于交易理念的预判）
        if 'trend_analysis' in context:
            trend = context['trend_analysis']
            bias_warning = "🚨 超过5%，严禁追高！" if trend.get('bias_ma5', 0) > 5 else "✅ 安全范围"
            prompt += f"""
### 趋势分析预判（基于交易理念）
| 指标 | 数值 | 判定 |
|------|------|------|
| 趋势状态 | {trend.get('trend_status', '未知')} | |
| 均线排列 | {trend.get('ma_alignment', '未知')} | MA5>MA10>MA20为多头 |
| 趋势强度 | {trend.get('trend_strength', 0)}/100 | |
| **乖离率(MA5)** | **{trend.get('bias_ma5', 0):+.2f}%** | {bias_warning} |
| 乖离率(MA10) | {trend.get('bias_ma10', 0):+.2f}% | |
| 量能状态 | {trend.get('volume_status', '未知')} | {trend.get('volume_trend', '')} |
| 系统信号 | {trend.get('buy_signal', '未知')} | |
| 系统评分 | {trend.get('signal_score', 0)}/100 | |

#### 系统分析理由
**买入理由**：
{chr(10).join('- ' + r for r in trend.get('signal_reasons', ['无'])) if trend.get('signal_reasons') else '- 无'}

**风险因素**：
{chr(10).join('- ' + r for r in trend.get('risk_factors', ['无'])) if trend.get('risk_factors') else '- 无'}
"""
        
        # 添加昨日对比数据
        if 'yesterday' in context:
            volume_change = context.get('volume_change_ratio', 'N/A')
            prompt += f"""
### 量价变化
- 成交量较昨日变化：{volume_change}倍
- 价格较昨日变化：{context.get('price_change_ratio', 'N/A')}%
"""
        
        # 添加新闻搜索结果（重点区域）
        prompt += """
---

## 📰 舆情情报
"""
        if news_context:
            prompt += f"""
以下是 **{stock_name}({code})** 近7日的新闻搜索结果，请重点提取：
1. 🚨 **风险警报**：减持、处罚、利空
2. 🎯 **利好催化**：业绩、合同、政策
3. 📊 **业绩预期**：年报预告、业绩快报

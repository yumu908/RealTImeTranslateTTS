package com.example.myapplication1.translation

import android.util.Log
import java.io.BufferedReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Post-translation refiner using a fast LLM API.
 *
 * Two modes:
 * 1. **Per-sentence** (`refine`): quick polish of a single translated sentence (legacy).
 * 2. **Paragraph-level** (`refineParagraph`): takes all (EN, rawZH) pairs in a
 *    paragraph, fixes truncations, merges fragments, and produces one coherent
 *    Chinese paragraph.  This is the primary mode used by the new pipeline.
 *
 * Designed for low latency: short timeouts, compact prompts.
 * If the API call fails or times out, the raw translation is returned unchanged.
 */
class TranslationRefiner(
    private val apiKey: String,
    private val baseUrl: String = "https://api.groq.com/openai/v1",
    private val model: String = "llama-3.3-70b-versatile",
    private val maxContextSize: Int = 20
) {
    companion object {
        private const val TAG = "TransRefiner"

        private val cloudClient = OkHttpClient.Builder()
            .connectTimeout(3, TimeUnit.SECONDS)
            .readTimeout(8, TimeUnit.SECONDS)
            .build()

        private val localClient = OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .build()

        // ---- Paragraph-level prompt (primary) ----
        private const val PARAGRAPH_REFINE_PROMPT =
            "你是专业同声传译后处理器。\n" +
            "以下是语音识别（ASR）逐句切分的英文原文及其逐句机器翻译。\n" +
            "请将这些翻译整合为一段连贯、自然的书面中文。\n\n" +
            "处理规则（按优先级）：\n" +
            "1. 【合并截断】ASR 常将一句完整话语切成多个短片段，请识别并合并\n" +
            "2. 【修正语法】改正语法错误、主谓搭配错误、量词错误\n" +
            "3. 【提升自然度】将逐字直译改为符合中文表达习惯的地道说法\n" +
            "4. 【保持原意】不增添、不删减关键信息\n" +
            "5. 【格式要求】只输出最终中文段落，不加编号、引号、注释或任何额外字符"

        /** Provider constants */
        const val PROVIDER_OFF = 0
        const val PROVIDER_GROQ = 1
        const val PROVIDER_OPENAI = 2
        const val PROVIDER_LOCAL = 3
        const val PROVIDER_ON_DEVICE = 4
        const val PROVIDER_CLAUDE = 5

        data class ModelPreset(val id: String, val label: String, val desc: String)

        val GROQ_MODELS = listOf(
            ModelPreset("llama-3.3-70b-versatile", "Llama 3.3 70B", "极速 · 免费额度"),
            ModelPreset("llama-3.1-8b-instant", "Llama 3.1 8B", "最快 · 免费"),
            ModelPreset("llama-3.1-70b-versatile", "Llama 3.1 70B", "高质量"),
            ModelPreset("gemma2-9b-it", "Gemma 2 9B", "Google · 免费"),
            ModelPreset("mixtral-8x7b-32768", "Mixtral 8x7B", "MoE · 均衡"),
            ModelPreset("deepseek-r1-distill-llama-70b", "DeepSeek R1 70B", "深度推理"),
        )

        val OPENAI_MODELS = listOf(
            ModelPreset("gpt-4.1-nano", "GPT-4.1 Nano", "最快 · 最低价"),
            ModelPreset("gpt-4.1-mini", "GPT-4.1 Mini", "快速 · 均衡"),
            ModelPreset("gpt-4.1", "GPT-4.1", "最新旗舰"),
            ModelPreset("gpt-4o-mini", "GPT-4o Mini", "快速 · 低价"),
            ModelPreset("gpt-4o", "GPT-4o", "高质量"),
            ModelPreset("o4-mini", "o4-mini", "推理模型 · 快速"),
            ModelPreset("o3-mini", "o3-mini", "推理模型"),
        )

        val CLAUDE_MODELS = listOf(
            ModelPreset("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "极速 · 推荐"),
            ModelPreset("claude-sonnet-4-20250514", "Claude Sonnet 4", "均衡 · 高质量"),
            ModelPreset("claude-opus-4-20250805", "Claude Opus 4", "最强推理"),
            ModelPreset("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet", "上代旗舰"),
            ModelPreset("claude-3-5-haiku-20241022", "Claude 3.5 Haiku", "上代极速"),
        )

        val LOCAL_MODELS = listOf(
            ModelPreset("qwen2.5:3b", "Qwen 2.5 3B", "极快 · 内存~2GB"),
            ModelPreset("qwen2.5:7b", "Qwen 2.5 7B", "均衡 · 内存~5GB"),
            ModelPreset("qwen2.5:14b", "Qwen 2.5 14B", "高质量 · 内存~9GB"),
            ModelPreset("llama3.2:3b", "Llama 3.2 3B", "极快 · 内存~2GB"),
            ModelPreset("gemma2:2b", "Gemma 2 2B", "最快 · 内存~1.5GB"),
            ModelPreset("phi3:mini", "Phi-3 Mini", "微软 · 内存~2GB"),
            ModelPreset("deepseek-r1:7b", "DeepSeek R1 7B", "推理 · 内存~5GB"),
        )

        val ON_DEVICE_MODELS = listOf(
            ModelPreset("qwen2.5:0.5b", "Qwen 2.5 0.5B", "最小 · 中文最佳 · ~350MB"),
            ModelPreset("qwen2.5:1.5b", "Qwen 2.5 1.5B", "推荐 · 中文优秀 · ~950MB"),
            ModelPreset("gemma2:2b", "Gemma 2 2B", "Google · ~1.5GB"),
            ModelPreset("llama3.2:1b", "Llama 3.2 1B", "Meta · 最快 · ~700MB"),
            ModelPreset("phi3:mini", "Phi-3 Mini 3.8B", "微软 · ~2.3GB"),
        )

        fun providerBaseUrl(provider: Int): String = when (provider) {
            PROVIDER_GROQ -> "https://api.groq.com/openai/v1"
            PROVIDER_OPENAI -> "https://api.openai.com/v1"
            PROVIDER_CLAUDE -> "https://api.anthropic.com/v1"
            PROVIDER_ON_DEVICE -> "http://localhost:11434/v1"
            else -> ""
        }

        fun defaultModel(provider: Int): String = when (provider) {
            PROVIDER_GROQ -> GROQ_MODELS.first().id
            PROVIDER_OPENAI -> OPENAI_MODELS.first().id
            PROVIDER_CLAUDE -> CLAUDE_MODELS.first().id
            PROVIDER_LOCAL -> LOCAL_MODELS.first().id
            PROVIDER_ON_DEVICE -> ON_DEVICE_MODELS.first().id
            else -> ""
        }
    }

    private val isLocal: Boolean = baseUrl.let {
        val lower = it.lowercase()
        lower.contains("localhost") || lower.contains("127.0.0.1") ||
                lower.contains("192.168.") || lower.contains("10.0.")
    }

    /** Whether this refiner targets the Anthropic Messages API (not OpenAI-compatible). */
    private val isClaude: Boolean = baseUrl.contains("anthropic.com")

    // Paragraph-level context: list of past refined paragraphs for coherence
    private val paragraphContext = ArrayDeque<Pair<String, String>>() // EN paragraph → ZH paragraph
    private val mutex = Mutex()

    // ===================== Paragraph-level refinement (primary) =====================

    /**
     * Refine a complete paragraph: takes all (EN, rawZH) sentence pairs and
     * produces one coherent Chinese paragraph.
     *
     * @param segments list of (English sentence, machine-translated Chinese) pairs
     * @return polished Chinese paragraph, or concatenated raw if API fails
     */
    suspend fun refineParagraph(segments: List<Pair<String, String>>): String {
        if (segments.isEmpty()) return ""
        if (segments.size == 1 && segments[0].second.isNotBlank()) {
            // Single sentence — still refine for grammar/completeness
            return refineSingle(segments[0].first, segments[0].second)
        }

        val ctx = mutex.withLock { paragraphContext.toList() }
        val rawFallback = segments.joinToString("") { it.second }

        return try {
            val userMsg = buildParagraphRefineMsg(segments, ctx)
            val result = callLlm(PARAGRAPH_REFINE_PROMPT, userMsg)

            // Sanity: result should be non-empty and not absurdly long
            if (result.isBlank() || result.length > rawFallback.length * 4) {
                Log.w(TAG, "Paragraph refinement suspicious, using raw")
                rawFallback
            } else {
                // Update context
                val combinedEn = segments.joinToString(" ") { it.first }
                mutex.withLock {
                    paragraphContext.addLast(combinedEn to result)
                    while (paragraphContext.size > maxContextSize) paragraphContext.removeFirst()
                }
                Log.d(TAG, "Paragraph refined: ${segments.size} segments → ${result.take(80)}")
                result
            }
        } catch (e: Throwable) {
            Log.w(TAG, "Paragraph refinement failed: ${e.message}")
            rawFallback
        }
    }

    /** Quick single-sentence refinement. */
    private suspend fun refineSingle(en: String, rawZh: String): String {
        val ctx = mutex.withLock { paragraphContext.toList() }
        return try {
            val sb = StringBuilder()
            if (ctx.isNotEmpty()) {
                sb.append("上下文:\n")
                ctx.takeLast(5).forEachIndexed { i, (e, z) ->
                    sb.append("${i + 1}. $e → $z\n")
                }
                sb.append("\n")
            }
            sb.append("原文: $en\n机翻: $rawZh")

            val result = callLlm(
                "你是同声传译后处理器。修正机器翻译使其自然通顺。只输出修正后的中文。",
                sb.toString()
            )
            if (result.isBlank() || result.length > rawZh.length * 3) rawZh else result
        } catch (e: Throwable) {
            rawZh
        }
    }

    suspend fun clearContext() {
        mutex.withLock { paragraphContext.clear() }
    }

    // ===================== Internal =====================

    private fun buildParagraphRefineMsg(
        segments: List<Pair<String, String>>,
        ctx: List<Pair<String, String>>
    ): String {
        val sb = StringBuilder()
        if (ctx.isNotEmpty()) {
            sb.append("前文（已翻译的段落，供参考上下文）:\n")
            ctx.takeLast(3).forEach { (e, z) ->
                sb.append("$e\n→ $z\n\n")
            }
        }
        sb.append("当前段落（需要整合润色）:\n")
        segments.forEachIndexed { i, (en, zh) ->
            sb.append("${i + 1}. \"$en\" → \"$zh\"\n")
        }
        return sb.toString()
    }

    private suspend fun callLlm(systemPrompt: String, userMsg: String): String =
        withContext(Dispatchers.IO) {
            val inputWords = userMsg.split(Regex("""\s+""")).size
            val estimatedTokens = (inputWords * 3).coerceIn(200, 2000)

            if (isClaude) {
                callClaudeLlm(systemPrompt, userMsg, estimatedTokens)
            } else {
                callOpenAiCompatibleLlm(systemPrompt, userMsg, estimatedTokens)
            }
        }

    private fun callOpenAiCompatibleLlm(systemPrompt: String, userMsg: String, maxTokens: Int): String {
        val json = JSONObject().apply {
            put("model", model)
            put("messages", JSONArray().apply {
                put(JSONObject().apply {
                    put("role", "system")
                    put("content", systemPrompt)
                })
                put(JSONObject().apply {
                    put("role", "user")
                    put("content", userMsg)
                })
            })
            put("temperature", 0.2)
            put("max_tokens", maxTokens)
            put("stream", true)
        }

        val request = Request.Builder()
            .url("${baseUrl.trimEnd('/')}/chat/completions")
            .addHeader("Authorization", "Bearer $apiKey")
            .post(json.toString().toRequestBody("application/json".toMediaType()))
            .build()

        val response = (if (isLocal) localClient else cloudClient).newCall(request).execute()
        if (!response.isSuccessful) {
            val errBody = response.body?.string() ?: ""
            throw Exception("HTTP ${response.code}: $errBody")
        }

        return collectSseTokens(response.body?.charStream()?.buffered())
    }

    private fun callClaudeLlm(systemPrompt: String, userMsg: String, maxTokens: Int): String {
        val json = JSONObject().apply {
            put("model", model)
            put("max_tokens", maxTokens)
            put("system", systemPrompt)
            put("messages", JSONArray().apply {
                put(JSONObject().apply {
                    put("role", "user")
                    put("content", userMsg)
                })
            })
            put("temperature", 0.2)
            put("stream", true)
        }

        val request = Request.Builder()
            .url("${baseUrl.trimEnd('/')}/messages")
            .addHeader("x-api-key", apiKey)
            .addHeader("anthropic-version", "2023-06-01")
            .post(json.toString().toRequestBody("application/json".toMediaType()))
            .build()

        val response = cloudClient.newCall(request).execute()
        if (!response.isSuccessful) {
            val errBody = response.body?.string() ?: ""
            throw Exception("HTTP ${response.code}: $errBody")
        }

        return collectClaudeSseTokens(response.body?.charStream()?.buffered())
    }
}

package com.example.myapplication1.translation

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import okhttp3.FormBody
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

abstract class TranslationEngine {
    abstract suspend fun translate(text: String): String

    /**
     * Context-aware translation. Engines that support background/glossary/latency
     * override this; others fall back to the plain [translate].
     */
    open suspend fun translate(text: String, context: TranslationContext): String = translate(text)

    /** True if this engine uses an LLM API (can combine translation + refinement in one call). */
    open val isLlmBased: Boolean = false
    open fun close() {}
}

/**
 * Shared system prompt for simultaneous interpreting.
 * Concise and deterministic: low temperature + brief output instruction.
 */
private val LANG_NAMES = mapOf(
    "en" to "英文", "zh" to "中文", "ja" to "日语", "ko" to "韩语",
    "fr" to "法语", "de" to "德语", "es" to "西班牙语", "ru" to "俄语",
    "pt" to "葡萄牙语", "it" to "意大利语", "ar" to "阿拉伯语",
)

/** Build dynamic system prompt based on source/target language pair. */
private fun buildSiSystemPrompt(src: String, tgt: String): String {
    val srcName = LANG_NAMES[src] ?: src
    val tgtName = LANG_NAMES[tgt] ?: tgt
    return "你是专业同声传译员（${srcName}译${tgtName}）。" +
        "请将用户输入的${srcName}直接翻译为地道、简洁的${tgtName}。" +
        "只输出翻译结果，不得包含任何解释、标注或额外标点。"
}

/**
 * MLKit offline translation (EN→ZH).
 */
class MlKitTranslation(
    private val translator: com.google.mlkit.nl.translate.Translator
) : TranslationEngine() {
    override suspend fun translate(text: String): String = suspendCancellableCoroutine { cont ->
        translator.translate(text)
            .addOnSuccessListener { cont.resume(it) }
            .addOnFailureListener { cont.resumeWithException(it) }
    }
}

/**
 * LLM-based translation via OpenAI-compatible chat completions API (streaming).
 *
 * Uses SSE streaming so the first tokens arrive as soon as the model starts generating,
 * reducing time-to-first-byte compared to waiting for the full non-streaming response.
 * The complete translated string is returned once the stream ends.
 *
 * Works with OpenAI, Groq, and other compatible providers.
 */
class LLMTranslation(
    private val apiKey: String,
    private val baseUrl: String = "https://api.openai.com/v1",
    private val model: String = "gpt-4o-mini"
) : TranslationEngine() {
    override val isLlmBased: Boolean = true

    companion object {
        private const val TAG = "LLMTranslation"

        private val sharedClient = OkHttpClient.Builder()
            .connectTimeout(8, TimeUnit.SECONDS)
            .readTimeout(20, TimeUnit.SECONDS)
            .build()

        /**
         * Estimate a safe max_tokens bound for a single-sentence translation.
         * Chinese is ~1.5x English word count; each Chinese char ≈ 2 BPE tokens → factor ≈ 3.
         * Upper bound is 600 (single sentence); use a higher cap for multi-sentence paragraph work.
         */
        private fun maxTokensFor(text: String): Int {
            val words = text.trim().split(Regex("""\s+""")).size
            return (words * 3).coerceIn(80, 600)
        }
    }

    override suspend fun translate(text: String): String = translate(text, TranslationContext())

    override suspend fun translate(text: String, context: TranslationContext): String = withContext(Dispatchers.IO) {
        val systemPrompt = buildContextPrompt(buildSiSystemPrompt(context.sourceLang, context.targetLang), context)

        val json = JSONObject().apply {
            put("model", model)
            put("messages", JSONArray().apply {
                put(JSONObject().apply { put("role", "system"); put("content", systemPrompt) })
                put(JSONObject().apply { put("role", "user"); put("content", text) })
            })
            put("temperature", 0.2)
            put("max_tokens", maxTokensFor(text))
            put("stream", true)
        }

        val request = Request.Builder()
            .url("${baseUrl.trimEnd('/')}/chat/completions")
            .addHeader("Authorization", "Bearer $apiKey")
            .post(json.toString().toRequestBody("application/json".toMediaType()))
            .build()

        val response = sharedClient.newCall(request).execute()
        if (!response.isSuccessful) {
            val errBody = response.body?.string() ?: ""
            throw Exception("HTTP ${response.code}: $errBody")
        }

        collectSseTokens(response.body?.charStream()?.buffered())
    }
}

/**
 * Local server translation (Ollama, LM Studio, etc.) via OpenAI-compatible API.
 * Uses streaming + longer timeouts for local inference.
 */
class LocalServerTranslation(
    private val serverUrl: String,
    private val model: String = "qwen2.5:7b"
) : TranslationEngine() {
    override val isLlmBased: Boolean = true

    companion object {
        private const val TAG = "LocalServerTranslation"

        private val localClient = OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .build()
    }

    override suspend fun translate(text: String): String = translate(text, TranslationContext())

    override suspend fun translate(text: String, context: TranslationContext): String = withContext(Dispatchers.IO) {
        val url = serverUrl.trimEnd('/') + "/chat/completions"
        val systemPrompt = buildContextPrompt(buildSiSystemPrompt(context.sourceLang, context.targetLang), context)

        val json = JSONObject().apply {
            put("model", model)
            put("messages", JSONArray().apply {
                put(JSONObject().apply { put("role", "system"); put("content", systemPrompt) })
                put(JSONObject().apply { put("role", "user"); put("content", text) })
            })
            put("temperature", 0.2)
            put("max_tokens", 600)
            put("stream", true)
        }

        val request = Request.Builder()
            .url(url)
            .post(json.toString().toRequestBody("application/json".toMediaType()))
            .build()

        val response = localClient.newCall(request).execute()
        if (!response.isSuccessful) {
            val errBody = response.body?.string() ?: ""
            throw Exception("HTTP ${response.code}: $errBody")
        }

        collectSseTokens(response.body?.charStream()?.buffered())
    }
}

/**
 * DeepL API translation.
 * Automatically detects free vs pro plan based on API key suffix.
 */
class DeepLTranslation(private val apiKey: String) : TranslationEngine() {

    companion object {
        private val sharedClient = OkHttpClient.Builder()
            .connectTimeout(8, TimeUnit.SECONDS)
            .readTimeout(15, TimeUnit.SECONDS)
            .build()
    }

    override suspend fun translate(text: String): String = translate(text, TranslationContext())

    override suspend fun translate(text: String, context: TranslationContext): String = withContext(Dispatchers.IO) {
        val base = if (apiKey.endsWith(":fx")) "https://api-free.deepl.com" else "https://api.deepl.com"
        val formBuilder = FormBody.Builder()
            .add("text", text)
            .add("source_lang", context.sourceLang.uppercase())
            .add("target_lang", deeplTargetCode(context.targetLang))

        // DeepL context parameter: additional context for disambiguation (not billed)
        if (context.background.isNotBlank()) {
            formBuilder.add("context", context.background.take(300))
        }

        // DeepL model_type: latency_optimized or quality_optimized
        when (context.latencyMode) {
            LatencyMode.REALTIME -> formBuilder.add("model_type", "latency_optimized")
            LatencyMode.QUALITY -> formBuilder.add("model_type", "quality_optimized")
            else -> {} // balanced uses DeepL default
        }

        val request = Request.Builder()
            .url("$base/v2/translate")
            .addHeader("Authorization", "DeepL-Auth-Key $apiKey")
            .post(formBuilder.build())
            .build()

        val response = sharedClient.newCall(request).execute()
        val body = response.body?.string() ?: throw Exception("Empty response")
        if (!response.isSuccessful) throw Exception("DeepL ${response.code}")

        JSONObject(body)
            .getJSONArray("translations")
            .getJSONObject(0)
            .getString("text")
            .trim()
    }
}

// ---------------------------------------------------------------------------
// Context prompt builder — shared by LLM engines
// ---------------------------------------------------------------------------

/**
 * Augment a base system prompt with optional background context and glossary terms.
 * Returns the original prompt unmodified if context has nothing to add.
 */
/** Map ISO 639-1 to DeepL target language codes (some require regional variants). */
private fun deeplTargetCode(lang: String): String = when (lang) {
    "en" -> "EN-US"; "pt" -> "PT-BR"; "zh" -> "ZH"; else -> lang.uppercase()
}

internal fun buildContextPrompt(base: String, context: TranslationContext): String {
    if (context.background.isBlank() && context.glossaryTerms.isEmpty()) return base
    return buildString {
        append(base)
        if (context.background.isNotBlank()) {
            append("\n\n背景信息（仅供理解上下文，无需翻译）：")
            append(context.background.take(300))
        }
        if (context.glossaryTerms.isNotEmpty()) {
            append("\n\n参考术语（请优先使用以下译法）：")
            context.glossaryTerms.entries.take(20).forEach { (en, zh) ->
                append("\n- $en → $zh")
            }
        }
    }
}

/**
 * Claude API translation via Anthropic Messages endpoint (streaming).
 *
 * Key differences from OpenAI-compatible:
 * - Endpoint: /v1/messages (not /chat/completions)
 * - Auth: x-api-key header (not Bearer token)
 * - Body: system is top-level string, not a system message
 * - SSE: content_block_delta events with delta.text (not choices[0].delta.content)
 */
class ClaudeTranslation(
    private val apiKey: String,
    private val model: String = "claude-sonnet-4-20250514"
) : TranslationEngine() {
    override val isLlmBased: Boolean = true

    companion object {
        private const val TAG = "ClaudeTranslation"
        private const val BASE_URL = "https://api.anthropic.com/v1/messages"
        private const val ANTHROPIC_VERSION = "2023-06-01"

        private val sharedClient = OkHttpClient.Builder()
            .connectTimeout(8, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .build()

        data class ModelPreset(val id: String, val label: String, val desc: String)

        val MODELS = listOf(
            ModelPreset("claude-sonnet-4-20250514", "Claude Sonnet 4", "均衡 · 推荐"),
            ModelPreset("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "极速 · 最低价"),
            ModelPreset("claude-opus-4-20250805", "Claude Opus 4", "最强推理 · 最贵"),
            ModelPreset("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet", "上代旗舰"),
            ModelPreset("claude-3-5-haiku-20241022", "Claude 3.5 Haiku", "上代极速"),
        )
    }

    override suspend fun translate(text: String): String = translate(text, TranslationContext())

    override suspend fun translate(text: String, context: TranslationContext): String = withContext(Dispatchers.IO) {
        val systemPrompt = buildContextPrompt(buildSiSystemPrompt(context.sourceLang, context.targetLang), context)
        val words = text.trim().split(Regex("""\s+""")).size
        val maxTokens = (words * 3).coerceIn(80, 600)

        val json = JSONObject().apply {
            put("model", model)
            put("max_tokens", maxTokens)
            put("system", systemPrompt)
            put("messages", JSONArray().apply {
                put(JSONObject().apply { put("role", "user"); put("content", text) })
            })
            put("temperature", 0.2)
            put("stream", true)
        }

        val request = Request.Builder()
            .url(BASE_URL)
            .addHeader("x-api-key", apiKey)
            .addHeader("anthropic-version", ANTHROPIC_VERSION)
            .addHeader("content-type", "application/json")
            .post(json.toString().toRequestBody("application/json".toMediaType()))
            .build()

        val response = sharedClient.newCall(request).execute()
        if (!response.isSuccessful) {
            val errBody = response.body?.string() ?: ""
            throw Exception("HTTP ${response.code}: $errBody")
        }

        collectClaudeSseTokens(response.body?.charStream()?.buffered())
    }
}

// ---------------------------------------------------------------------------
// SSE streaming helper — shared by LLMTranslation and LocalServerTranslation
// ---------------------------------------------------------------------------

/**
 * Collects all delta tokens from an OpenAI-compatible SSE stream and returns
 * the concatenated result.  Handles both `data: {...}` lines and the `[DONE]`
 * sentinel.  Returns an empty string on parse errors rather than throwing.
 */
internal fun collectSseTokens(reader: BufferedReader?): String {
    if (reader == null) return ""
    val sb = StringBuilder()
    try {
        reader.use { br ->
            br.lineSequence().forEach { line ->
                if (!line.startsWith("data:")) return@forEach
                val payload = line.removePrefix("data:").trim()
                if (payload == "[DONE]") return@forEach
                try {
                    val delta = JSONObject(payload)
                        .getJSONArray("choices")
                        .getJSONObject(0)
                        .getJSONObject("delta")
                    val content = delta.optString("content", "")
                    if (content.isNotEmpty()) sb.append(content)
                } catch (_: Exception) { /* skip malformed lines */ }
            }
        }
    } catch (e: Exception) {
        Log.w("SSE", "Stream read error: ${e.message}")
    }
    return sb.toString().trim()
}

/**
 * Collects text tokens from a Claude Messages SSE stream.
 * Claude uses `event: content_block_delta` lines followed by `data: {...}` lines
 * where the text lives at `delta.text`.
 */
internal fun collectClaudeSseTokens(reader: BufferedReader?): String {
    if (reader == null) return ""
    val sb = StringBuilder()
    try {
        reader.use { br ->
            br.lineSequence().forEach { line ->
                if (!line.startsWith("data:")) return@forEach
                val payload = line.removePrefix("data:").trim()
                if (payload.isEmpty()) return@forEach
                try {
                    val obj = JSONObject(payload)
                    val type = obj.optString("type")
                    if (type == "content_block_delta") {
                        val text = obj.optJSONObject("delta")?.optString("text", "") ?: ""
                        if (text.isNotEmpty()) sb.append(text)
                    }
                } catch (_: Exception) { /* skip malformed lines */ }
            }
        }
    } catch (e: Exception) {
        Log.w("ClaudeSSE", "Stream read error: ${e.message}")
    }
    return sb.toString().trim()
}

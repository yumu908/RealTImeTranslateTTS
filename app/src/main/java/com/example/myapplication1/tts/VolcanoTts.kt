package com.example.myapplication1.tts

import android.media.AudioAttributes
import android.media.MediaPlayer
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.UUID
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume

/**
 * 火山引擎（字节跳动）语音合成 API 客户端。
 *
 * API: POST https://openspeech.bytedance.com/api/v1/tts
 * 认证: Bearer;<access_token> （注意授权头用分号分隔，火山引擎专有格式）
 * 请求/响应体: JSON，响应中 data 字段为 base64 编码的音频。
 *
 * 参考: https://www.volcengine.com/docs/6561/79817
 */
class VolcanoTts(
    private val appId: String,
    private val accessToken: String,
    private val cluster: String,
    private val cacheDir: File,
) : AutoCloseable {

    companion object {
        private const val TAG = "VolcanoTts"
        private const val URL = "https://openspeech.bytedance.com/api/v1/tts"

        /**
         * 火山引擎常用音色（中文·普通音色 voice_type）。
         * voice_type 值来自官方文档；声音模型集群一般为 "volcano_tts"。
         */
        val ZH_VOICES = listOf(
            "BV700_streaming" to "灿灿 (女·亲和)",
            "BV701_streaming" to "擎苍 (男·沉稳)",
            "BV002_streaming" to "通用女声",
            "BV001_streaming" to "通用男声",
            "BV102_streaming" to "范儿 (女·知性)",
            "BV119_streaming" to "通用男声·预置",
            "BV405_streaming" to "阳光青年",
            "BV406_streaming" to "活力女孩",
            "BV407_streaming" to "和蔼青年",
            "BV408_streaming" to "温柔小哥",
            "BV503_streaming" to "知性女声",
            "BV504_streaming" to "解说小哥",
        )

        val EN_VOICES = listOf(
            "en_male_adam_mars_bigtts" to "Adam (男·美式)",
            "en_female_anna_mars_bigtts" to "Anna (女·美式)",
            "en_male_smith_mars_bigtts" to "Smith (男·英式)",
            "en_female_sarah_mars_bigtts" to "Sarah (女·英式)",
        )

        val JA_VOICES = listOf(
            "multi_male_jingqiangkanai_moon_bigtts" to "通用男声 (日)",
            "multi_female_shuangkuaisisi_moon_bigtts" to "通用女声 (日)",
        )

        /** 根据语言返回推荐音色列表，未命中则回落到中文。 */
        fun voicesForLang(lang: String): List<Pair<String, String>> = when (lang) {
            "zh" -> ZH_VOICES
            "en" -> EN_VOICES
            "ja" -> JA_VOICES
            else -> ZH_VOICES
        }

        /** 默认集群（新建的火山引擎应用大多属于此集群）。 */
        const val DEFAULT_CLUSTER = "volcano_tts"
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    @Volatile private var currentPlayer: MediaPlayer? = null

    /**
     * Synthesize and play.
     * @param speed 0.2 .. 3.0 (1.0 = normal)
     */
    suspend fun speak(
        text: String,
        voice: String = "BV700_streaming",
        speed: Float = 1.0f,
    ) {
        if (text.isBlank() || appId.isBlank() || accessToken.isBlank()) return
        val file = synthesize(text, voice, speed) ?: return
        try { playFile(file) } finally { file.delete() }
    }

    private suspend fun synthesize(text: String, voice: String, speed: Float): File? =
        withContext(Dispatchers.IO) {
            try {
                val reqId = UUID.randomUUID().toString()
                val body = JSONObject().apply {
                    put("app", JSONObject().apply {
                        put("appid", appId)
                        put("token", accessToken)
                        put("cluster", cluster.ifBlank { DEFAULT_CLUSTER })
                    })
                    put("user", JSONObject().apply {
                        put("uid", "vri_${System.currentTimeMillis()}")
                    })
                    put("audio", JSONObject().apply {
                        put("voice_type", voice)
                        put("encoding", "mp3")
                        put("speed_ratio", speed.coerceIn(0.2f, 3.0f).toDouble())
                        put("volume_ratio", 1.0)
                        put("pitch_ratio", 1.0)
                    })
                    put("request", JSONObject().apply {
                        put("reqid", reqId)
                        put("text", text)
                        put("text_type", "plain")
                        put("operation", "query")
                    })
                }

                val request = Request.Builder()
                    .url(URL)
                    // 火山引擎要求 "Bearer;<token>" 这种奇怪的分号分隔格式。
                    .addHeader("Authorization", "Bearer;$accessToken")
                    .post(body.toString().toRequestBody("application/json".toMediaType()))
                    .build()

                val response = client.newCall(request).execute()
                val respBody = response.body?.string()
                if (!response.isSuccessful || respBody == null) {
                    Log.e(TAG, "HTTP ${response.code}: ${respBody?.take(200)}")
                    return@withContext null
                }

                val json = JSONObject(respBody)
                val code = json.optInt("code", -1)
                if (code != 3000) {
                    // 3000 = success, 其他都是错误
                    Log.e(TAG, "API code=$code message=${json.optString("message")}")
                    return@withContext null
                }
                val b64 = json.optString("data")
                if (b64.isBlank()) {
                    Log.e(TAG, "Response missing audio data field")
                    return@withContext null
                }

                val audio = try {
                    android.util.Base64.decode(b64, android.util.Base64.DEFAULT)
                } catch (e: Throwable) {
                    Log.e(TAG, "Base64 decode failed: ${e.message}")
                    return@withContext null
                }

                val outFile = File(cacheDir, "volcano_tts_${System.currentTimeMillis()}.mp3")
                FileOutputStream(outFile).use { it.write(audio) }
                outFile
            } catch (e: Throwable) {
                Log.e(TAG, "Synthesis failed: ${e.message}")
                null
            }
        }

    private suspend fun playFile(file: File) {
        stopPlayer()
        withContext(Dispatchers.IO) {
            suspendCancellableCoroutine { cont ->
                val mp = MediaPlayer()
                currentPlayer = mp
                try {
                    mp.setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_ASSISTANT)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()
                    )
                    mp.setDataSource(file.absolutePath)
                    mp.setOnPreparedListener { it.start() }
                    mp.setOnCompletionListener {
                        it.release(); if (currentPlayer === it) currentPlayer = null
                        if (cont.isActive) cont.resume(Unit)
                    }
                    mp.setOnErrorListener { mp2, _, _ ->
                        mp2.release(); if (currentPlayer === mp2) currentPlayer = null
                        if (cont.isActive) cont.resume(Unit)
                        true
                    }
                    mp.prepareAsync()
                    cont.invokeOnCancellation {
                        try { mp.release() } catch (_: Throwable) {}
                        if (currentPlayer === mp) currentPlayer = null
                    }
                } catch (t: Throwable) {
                    try { mp.release() } catch (_: Throwable) {}
                    if (currentPlayer === mp) currentPlayer = null
                    if (cont.isActive) cont.resume(Unit)
                }
            }
        }
    }

    fun stopPlayer() {
        try { currentPlayer?.let { if (it.isPlaying) it.stop(); it.release() } } catch (_: Throwable) {}
        currentPlayer = null
    }

    override fun close() {
        stopPlayer()
        client.dispatcher.executorService.shutdown()
    }
}

package com.example.myapplication1

import android.annotation.SuppressLint
import android.app.*
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import android.speech.tts.TextToSpeech
import android.util.Log
import android.util.TypedValue
import android.view.*
import android.widget.*
import com.example.myapplication1.translation.*
import com.example.myapplication1.translation.TranslationMeta
import com.example.myapplication1.translation.TranslationContext
import com.example.myapplication1.translation.LatencyMode
import com.example.myapplication1.translation.GlossaryManager
import com.example.myapplication1.tts.EdgeTts
import com.example.myapplication1.tts.GoogleTranslateTts
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.Translation
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max

class FloatingTranslateService : Service() {

    companion object {
        private const val TAG = "FloatingService"
        private const val CHANNEL_ID = "floating_translate"
        private const val NOTIF_ID = 2001

        @Volatile var isRunning = false
        /** Auto-launched from onPause — will auto-stop on resume */
        @Volatile var autoLaunched = false
        /** Set by MainActivity.onPause when app was recording */
        @Volatile var appWasRecording = false

        /** Callback to push translations back to MainActivity */
        var translationCallback: TranslationCallback? = null

        interface TranslationCallback {
            fun onFloatingTranslation(en: String, zh: String)
            fun onRecordingStateChanged(isRecording: Boolean) {}
        }
    }

    private var windowManager: WindowManager? = null
    private var floatingView: View? = null
    private var ttsConsumerJob: Job? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private val ttsQueue = Channel<String>(Channel.UNLIMITED)

    // ASR
    private var voskModel: Model? = null
    private var recognizer: Recognizer? = null
    private var audioRecord: AudioRecord? = null
    private var aec: android.media.audiofx.AcousticEchoCanceler? = null
    private var asrJob: Job? = null
    @Volatile private var recording = false
    @Volatile private var mediaCaptureSyncMode = false
    @Volatile private var mediaCaptureTranslateEnabled = false
    private var prevAsrCallback: MediaCaptureService.Companion.AsrCallback? = null

    // TTS
    private var systemTts: TextToSpeech? = null
    private var edgeTts: EdgeTts? = null
    private val googleTts = GoogleTranslateTts()
    private var volcanoTts: com.example.myapplication1.tts.VolcanoTts? = null
    private var volcanoVoiceIdx = 0
    private var ttsEngine = 0
    private var edgeVoiceIdx = 0
    private var autoSpeak = true
    private val ttsPendingCount = AtomicInteger(0)
    @Volatile private var isTtsSpeaking = false
    private var ttsSpeakEndTime = 0L
    @Volatile private var lastTtsLength = 0
    private val TTS_ECHO_GRACE_BASE_MS = 200L
    private val TTS_ECHO_GRACE_PER_CHAR_MS = 15L
    private fun ttsEchoGraceMs(): Long =
        TTS_ECHO_GRACE_BASE_MS + (lastTtsLength * TTS_ECHO_GRACE_PER_CHAR_MS).coerceAtMost(1500L)

    // Refiner
    private var translationRefiner: TranslationRefiner? = null

    // Translator
    private val mlkitTranslator by lazy {
        Translation.getClient(
            com.google.mlkit.nl.translate.TranslatorOptions.Builder()
                .setSourceLanguage(TranslateLanguage.ENGLISH)
                .setTargetLanguage(TranslateLanguage.CHINESE).build()
        )
    }
    private var translationEngine: TranslationEngine? = null
    private var translationEngineType = 0

    // History
    private var translationHistory: TranslationHistory? = null

    // Smart filter (cached from SharedPreferences at load time)
    private var smartFilterEnabled = true
    private var filterConfig = AsrTextFilter.FilterConfig()

    // Translation pipeline with ordered delivery
    private lateinit var translationPipeline: TranslationPipeline
    private var floatingQualityEngine: TranslationEngine? = null
    private var floatingLatencyMode = 0
    private var floatingBackground = ""
    private var floatingDomainHint = "auto"

    // UI refs
    private var tvPartial: TextView? = null
    private var tvTranslation: TextView? = null
    private var btnMic: ImageView? = null
    private var containerCard: View? = null
    private var expanded = true

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        isRunning = true
        createChannel()
        startForeground(NOTIF_ID, buildNotification())
        loadPrefs()
        edgeTts = EdgeTts(cacheDir)
        translationHistory = TranslationHistory(filesDir)
        translationHistory?.load()
        translationHistory?.continueOrNew()
        // Initialize glossary system (idempotent — safe if MainActivity already called it)
        GlossaryManager.init(this)
        // Initialize translation pipeline with ordered delivery
        translationPipeline = TranslationPipeline(scope)
        translationPipeline.setCallback(floatingPipelineCallback)
        translationPipeline.setEngine(translationEngine ?: MlKitTranslation(mlkitTranslator))
        translationPipeline.setQualityEngine(floatingQualityEngine)
        translationPipeline.setRefiner(translationRefiner)
        translationPipeline.translationContext = TranslationContext(
            background = floatingBackground,
            domainHint = floatingDomainHint,
            latencyMode = when (floatingLatencyMode) {
                1 -> LatencyMode.BALANCED
                2 -> LatencyMode.QUALITY
                else -> LatencyMode.REALTIME
            }
        )
        initTts()
        initVosk()
        startTtsConsumer()
        createFloatingWindow()
        if (MediaCaptureService.isCapturing) {
            setupMediaCaptureSync()
        } else if (appWasRecording) {
            // App was recording when it went to background — continue recording here
            scope.launch {
                delay(500) // Wait for Vosk to initialize
                startRecording()
            }
        }
    }

    override fun onDestroy() {
        isRunning = false
        teardownMediaCaptureSync()
        stopRecording()
        // Close current paragraph to trigger refinement before shutdown
        if (::translationPipeline.isInitialized) translationPipeline.closeParagraph(floatingParagraphId)
        ttsConsumerJob?.cancel(); ttsQueue.close()
        if (::translationPipeline.isInitialized) translationPipeline.close()
        scope.cancel()
        try { recognizer?.close() } catch (_: Throwable) {}
        try { voskModel?.close() } catch (_: Throwable) {}
        try { systemTts?.stop(); systemTts?.shutdown() } catch (_: Throwable) {}
        edgeTts?.close(); googleTts.close(); volcanoTts?.close()
        translationEngine?.close()
        floatingQualityEngine?.close()
        translationHistory?.close()
        try { mlkitTranslator.close() } catch (_: Throwable) {}
        try { windowManager?.removeView(floatingView) } catch (_: Throwable) {}
        super.onDestroy()
    }

    private fun loadPrefs() {
        val p = getSharedPreferences("vri_settings", Context.MODE_PRIVATE)
        ttsEngine = p.getInt("tts_engine", 0)
        edgeVoiceIdx = p.getInt("edge_voice_idx", 0)
        volcanoVoiceIdx = p.getInt("volcano_voice_idx", 0)
        autoSpeak = p.getBoolean("auto_speak", true)
        // Build Volcano TTS lazily only if credentials are present
        val vAppId = p.getString("volcano_app_id", "") ?: ""
        val vToken = p.getString("volcano_access_token", "") ?: ""
        val vCluster = p.getString("volcano_cluster", com.example.myapplication1.tts.VolcanoTts.DEFAULT_CLUSTER)
            ?: com.example.myapplication1.tts.VolcanoTts.DEFAULT_CLUSTER
        if (vAppId.isNotBlank() && vToken.isNotBlank()) {
            volcanoTts = com.example.myapplication1.tts.VolcanoTts(vAppId, vToken, vCluster, cacheDir)
        }
        translationEngineType = p.getInt("translation_engine", 0)
        translationEngine = when (translationEngineType) {
            1 -> LLMTranslation(p.getString("openai_key", "") ?: "")
            2 -> LLMTranslation(p.getString("groq_key", "") ?: "", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile")
            3 -> DeepLTranslation(p.getString("deepl_key", "") ?: "")
            4 -> LocalServerTranslation(
                p.getString("local_server_url", "http://192.168.1.100:11434/v1") ?: "",
                p.getString("local_server_model", "qwen2.5:7b") ?: "qwen2.5:7b"
            )
            5 -> {
                val engine = OnDeviceTranslation(this, OnDeviceTranslationModel.OPUS_MT_EN_ZH)
                engine.also { if (OnDeviceTranslationModel.OPUS_MT_EN_ZH.isDownloaded(this)) it.init() }
            }
            6 -> {
                val engine = OnDeviceTranslation(this, OnDeviceTranslationModel.NLLB_600M_INT8)
                engine.also { if (OnDeviceTranslationModel.NLLB_600M_INT8.isDownloaded(this)) it.init() }
            }
            7 -> ClaudeTranslation(
                p.getString("claude_key", "") ?: "",
                p.getString("claude_trans_model", "claude-sonnet-4-20250514") ?: "claude-sonnet-4-20250514"
            )
            else -> MlKitTranslation(mlkitTranslator)
        }

        // Build refiner
        val refineProvider = p.getInt("refine_provider", 0)
        val refineModel = p.getString("refine_model", "llama-3.3-70b-versatile") ?: "llama-3.3-70b-versatile"
        val refineServerUrl = p.getString("refine_server_url", "http://192.168.1.100:11434/v1") ?: "http://192.168.1.100:11434/v1"
        translationRefiner = when (refineProvider) {
            TranslationRefiner.PROVIDER_GROQ -> {
                val key = p.getString("groq_key", "") ?: ""
                if (key.isNotBlank()) TranslationRefiner(key, TranslationRefiner.providerBaseUrl(TranslationRefiner.PROVIDER_GROQ), refineModel) else null
            }
            TranslationRefiner.PROVIDER_OPENAI -> {
                val key = p.getString("openai_key", "") ?: ""
                if (key.isNotBlank()) TranslationRefiner(key, TranslationRefiner.providerBaseUrl(TranslationRefiner.PROVIDER_OPENAI), refineModel) else null
            }
            TranslationRefiner.PROVIDER_ON_DEVICE -> {
                TranslationRefiner("", TranslationRefiner.providerBaseUrl(TranslationRefiner.PROVIDER_ON_DEVICE), refineModel)
            }
            TranslationRefiner.PROVIDER_LOCAL -> {
                if (refineServerUrl.isNotBlank()) TranslationRefiner("", refineServerUrl, refineModel) else null
            }
            TranslationRefiner.PROVIDER_CLAUDE -> {
                val key = p.getString("claude_key", "") ?: ""
                if (key.isNotBlank()) TranslationRefiner(key, TranslationRefiner.providerBaseUrl(TranslationRefiner.PROVIDER_CLAUDE), refineModel) else null
            }
            else -> null
        }

        // Build quality engine for SWR dual-channel (same logic as MainActivity)
        val latencyMode = p.getInt("latency_mode", 0)
        if (latencyMode > 0) {
            val openaiKey = p.getString("openai_key", "") ?: ""
            val groqKey = p.getString("groq_key", "") ?: ""
            val deeplKey = p.getString("deepl_key", "") ?: ""
            val localUrl = p.getString("local_server_url", "") ?: ""
            val localModel = p.getString("local_server_model", "qwen2.5:7b") ?: "qwen2.5:7b"
            floatingQualityEngine = when {
                translationEngineType != 1 && openaiKey.isNotBlank() -> LLMTranslation(openaiKey)
                translationEngineType != 3 && deeplKey.isNotBlank() -> DeepLTranslation(deeplKey)
                translationEngineType != 2 && groqKey.isNotBlank() -> LLMTranslation(groqKey, "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile")
                translationEngineType != 4 && localUrl.isNotBlank() -> LocalServerTranslation(localUrl, localModel)
                else -> null
            }
            floatingLatencyMode = latencyMode
        }
        floatingBackground = p.getString("background_text", "") ?: ""
        floatingDomainHint = p.getString("domain_hint", "auto") ?: "auto"

        // Cache smart filter settings
        smartFilterEnabled = p.getBoolean("smart_filter_enabled", true)
        filterConfig = AsrTextFilter.FilterConfig(
            filterFillers = p.getBoolean("filter_fillers", true),
            filterEcho = p.getBoolean("filter_echo", true),
            filterNoise = p.getBoolean("filter_noise", true),
            filterMusic = p.getBoolean("filter_music", true)
        )
    }

    private fun createChannel() {
        val ch = NotificationChannel(CHANNEL_ID, "悬浮翻译", NotificationManager.IMPORTANCE_LOW)
        ch.description = "悬浮窗翻译服务"
        getSystemService(NotificationManager::class.java).createNotificationChannel(ch)
    }

    private fun buildNotification(): Notification {
        val pi = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val stopIntent = PendingIntent.getService(this, 1,
            Intent(this, FloatingTranslateService::class.java).setAction("STOP"),
            PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("悬浮翻译运行中")
            .setContentText("点击返回应用")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(pi)
            .addAction(Notification.Action.Builder(null, "停止", stopIntent).build())
            .setOngoing(true).build()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == "STOP") { stopSelf(); return START_NOT_STICKY }
        return START_STICKY
    }

    // ===================== Vosk =====================

    private fun initVosk() {
        scope.launch(Dispatchers.IO) {
            try {
                val dst = File(filesDir, "model-en")
                if (!dst.exists()) copyAssetDir("model-en", dst)
                voskModel = Model(dst.absolutePath)
                withContext(Dispatchers.Main) { Log.i(TAG, "Vosk ready") }
            } catch (e: Throwable) { Log.e(TAG, "Vosk: ${e.message}") }
        }
    }

    private fun setupMediaCaptureSync() {
        mediaCaptureSyncMode = true
        mediaCaptureTranslateEnabled = false
        stopRecording()
        prevAsrCallback = MediaCaptureService.asrCallback
        MediaCaptureService.asrCallback = object : MediaCaptureService.Companion.AsrCallback {
            override fun onPartial(text: String) {
                prevAsrCallback?.onPartial(text)
                // Always show partial text so user can see what's being heard
                updatePartial(text)
            }
            override fun onResult(text: String) {
                prevAsrCallback?.onResult(text)
                // Only translate if user explicitly enabled it
                if (mediaCaptureTranslateEnabled && text.isNotBlank()) {
                    onAsrResult(text)
                }
            }
            override fun onStateChanged(capturing: Boolean, status: String, sourceApp: String) {
                prevAsrCallback?.onStateChanged(capturing, status, sourceApp)
                if (!capturing) {
                    mediaCaptureSyncMode = false
                    mediaCaptureTranslateEnabled = false
                    scope.launch(Dispatchers.Main) {
                        updateMicIcon(false)
                        updatePartial("")
                    }
                }
            }
            override fun onError(msg: String) {
                prevAsrCallback?.onError(msg)
            }
        }
        updateMicIcon(false)
        updatePartial("媒体捕获中 · 点击麦克风开始翻译")
        Log.i(TAG, "Media capture sync mode (translate disabled, waiting for user)")
    }

    private fun teardownMediaCaptureSync() {
        if (mediaCaptureSyncMode) {
            MediaCaptureService.asrCallback = prevAsrCallback
            prevAsrCallback = null
            mediaCaptureSyncMode = false
        }
    }

    private fun initTts() {
        systemTts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                systemTts?.setLanguage(Locale.SIMPLIFIED_CHINESE)
                systemTts?.setAudioAttributes(android.media.AudioAttributes.Builder()
                    .setUsage(android.media.AudioAttributes.USAGE_ASSISTANT)
                    .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build())
            }
        }
        mlkitTranslator.downloadModelIfNeeded(com.google.mlkit.common.model.DownloadConditions.Builder().build())
    }

    @SuppressLint("MissingPermission")
    private fun startRecording() {
        if (mediaCaptureSyncMode) return
        if (recording || voskModel == null) return
        val sr = 16000
        val bs = max(AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT), 4096)
        try {
            audioRecord = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bs)
            try {
                if (android.media.audiofx.AcousticEchoCanceler.isAvailable()) {
                    aec = android.media.audiofx.AcousticEchoCanceler.create(audioRecord!!.audioSessionId)
                    aec?.enabled = true
                }
            } catch (_: Throwable) {}
            audioRecord?.startRecording()
        } catch (e: Throwable) { updatePartial("录音失败"); return }
        recognizer?.close()
        val localRec = Recognizer(voskModel, sr.toFloat())
        recognizer = localRec
        val localAudio = audioRecord ?: return
        recording = true
        updateMicIcon(true)
        updatePartial("正在听…")
        translationCallback?.onRecordingStateChanged(true)
        asrJob = scope.launch(Dispatchers.IO) {
            val buf = ShortArray(2048)
            while (isActive && recording) {
                val n = try { localAudio.read(buf, 0, buf.size) } catch (_: Throwable) { break }
                if (n <= 0) { if (n < 0) break else continue }
                try {
                    if (localRec.acceptWaveForm(buf, n)) {
                        val t = JSONObject(localRec.result).optString("text").trim()
                        if (t.isNotBlank()) withContext(Dispatchers.Main) { onAsrResult(t) }
                    } else {
                        val p = JSONObject(localRec.partialResult).optString("partial").trim()
                        if (p.isNotBlank()) withContext(Dispatchers.Main) {
                            updatePartial(p)
                        }
                    }
                } catch (_: Throwable) {}
            }
        }
    }

    private fun stopRecording() {
        if (!recording) return; recording = false
        asrJob?.cancel(); asrJob = null
        val r = recognizer?.finalResult?.let { JSONObject(it).optString("text") }.orEmpty().trim()
        if (r.isNotBlank()) onAsrResult(r)
        try { aec?.release() } catch (_: Throwable) {}; aec = null
        try { audioRecord?.stop() } catch (_: Throwable) {}
        try { audioRecord?.release() } catch (_: Throwable) {}
        audioRecord = null
        updateMicIcon(false); updatePartial("")
        translationCallback?.onRecordingStateChanged(false)
    }

    private fun onAsrResult(text: String) {
        // TTS回声抑制 — 动态宽限期：TTS文本越长回声越持久
        if (isTtsSpeaking || System.currentTimeMillis() - ttsSpeakEndTime < ttsEchoGraceMs()) return

        if (smartFilterEnabled) {
            val filtered = AsrTextFilter.filter(text, filterConfig) ?: return
            AsrTextFilter.recordText(filtered)
            return onAsrResultFiltered(filtered)
        }
        return onAsrResultFiltered(text)
    }

    /** Trust ASR boundaries — send each utterance directly to pipeline. */
    private fun onAsrResultFiltered(text: String) {
        updatePartial("")
        onSentence(text)
    }

    private var floatingParagraphId = 0

    private val floatingPipelineCallback = object : TranslationPipeline.Callback {
        override fun onTranslationStarted(seqId: Int, en: String) {
            updateTranslation("$en\n→ 翻译中…")
        }
        override fun onTranslationResult(seqId: Int, en: String, zh: String) {
            updateTranslation("$en\n→ $zh")
            translationHistory?.updateZhBySeqId(seqId, zh)
            translationCallback?.onFloatingTranslation(en, zh)
        }
        override fun onTranslationError(seqId: Int, en: String, error: String) {
            updateTranslation("$en\n→ [翻译失败]")
        }
        override fun onTtsReady(zh: String) {
            ttsPendingCount.incrementAndGet()
            ttsQueue.trySend(zh)
        }
        override fun onLatencyMeasured(translationMs: Long) {}
        override fun onCacheHit(seqId: Int) {}
        override fun onParagraphRefined(paragraphId: Int, refinedZh: String) {
            if (refinedZh.isNotBlank()) updateTranslation("→ $refinedZh")
        }
        override fun onTranslationUpgraded(seqId: Int, en: String, zh: String, meta: TranslationMeta) {
            updateTranslation("$en\n→ $zh ✓")
            translationHistory?.upsertZhBySeqId(seqId, zh)
        }
    }

    private fun onSentence(en: String) {
        val seqId = translationPipeline.allocateSeqId()
        translationHistory?.appendPending(seqId, en)
        translationPipeline.submitSentence(seqId, floatingParagraphId, en)
    }

    // ===================== TTS =====================

    private fun startTtsConsumer() {
        ttsConsumerJob = scope.launch {
            for (text in ttsQueue) {
                if (autoSpeak) {
                    // Long sentence protection: split at sentence boundaries if too long
                    if (text.length > 200) {
                        val parts = text.chunked(200)
                        for (part in parts) speakZh(part)
                    } else {
                        speakZh(text)
                    }
                }
                ttsPendingCount.decrementAndGet()

                // Max cumulative queue protection: if queue is too deep, drain excess
                if (ttsPendingCount.get() > 10) {
                    var drained = 0
                    while (ttsPendingCount.get() > 3) {
                        ttsQueue.tryReceive().getOrNull() ?: break
                        ttsPendingCount.decrementAndGet()
                        drained++
                    }
                    if (drained > 0) Log.d(TAG, "TTS queue overflow: drained $drained items")
                }
            }
        }
    }

    private fun dynamicEdgeRate(): String {
        val p = ttsPendingCount.get()
        val pct = when { p <= 1 -> 0; p == 2 -> 15; p == 3 -> 25; p == 4 -> 35; else -> 50 }
        return "+${pct}%"
    }

    private fun dynamicSystemRate(): Float {
        val p = ttsPendingCount.get()
        return when { p <= 1 -> 1.0f; p == 2 -> 1.15f; p == 3 -> 1.25f; p == 4 -> 1.35f; else -> 1.5f }
    }

    private suspend fun speakZh(text: String) {
        isTtsSpeaking = true
        lastTtsLength = text.length
        try { speakZhInternal(text) } finally {
            isTtsSpeaking = false
            ttsSpeakEndTime = System.currentTimeMillis()
        }
    }

    private suspend fun speakZhInternal(text: String) {
        when (ttsEngine) {
            0 -> {
                val v = EdgeTts.ZH_VOICES.getOrNull(edgeVoiceIdx)?.first ?: "zh-CN-XiaoxiaoNeural"
                try { edgeTts?.speak(text, voice = v, rate = dynamicEdgeRate()) } catch (_: Throwable) { speakSystem(text) }
            }
            1 -> speakSystem(text)
            2 -> try { googleTts.speak(text, "zh-CN") } catch (_: Throwable) {}
            3 -> {
                // OpenAI TTS — fall back to Edge if no key
                val v = EdgeTts.ZH_VOICES.getOrNull(edgeVoiceIdx)?.first ?: "zh-CN-XiaoxiaoNeural"
                try { edgeTts?.speak(text, voice = v, rate = dynamicEdgeRate()) } catch (_: Throwable) { speakSystem(text) }
            }
            5 -> {
                // 火山引擎 TTS — 未配置或失败时回退 Edge
                val vt = volcanoTts
                if (vt == null) {
                    val v = EdgeTts.ZH_VOICES.getOrNull(edgeVoiceIdx)?.first ?: "zh-CN-XiaoxiaoNeural"
                    try { edgeTts?.speak(text, voice = v, rate = dynamicEdgeRate()) } catch (_: Throwable) { speakSystem(text) }
                } else {
                    val voices = com.example.myapplication1.tts.VolcanoTts.ZH_VOICES
                    val vv = voices.getOrNull(volcanoVoiceIdx)?.first ?: voices.first().first
                    try { vt.speak(text, voice = vv) } catch (_: Throwable) { speakSystem(text) }
                }
            }
        }
    }

    private suspend fun speakSystem(text: String) {
        val rate = dynamicSystemRate()
        systemTts?.setSpeechRate(rate)
        systemTts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "f-${System.currentTimeMillis()}")
        delay((text.length * 280L / rate).toLong().coerceIn(500, 12000))
    }

    // ===================== Floating Window =====================

    @SuppressLint("ClickableViewAccessibility")
    private fun createFloatingWindow() {
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager

        val dp = { v: Float -> TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics).toInt() }

        val root = FrameLayout(this).apply {
            setPadding(dp(4f), dp(4f), dp(4f), dp(4f))
        }

        val bubble = ImageView(this).apply {
            setImageResource(android.R.drawable.ic_btn_speak_now)
            val bg = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(0xFF5B5FC7.toInt()) }
            background = bg; setPadding(dp(12f), dp(12f), dp(12f), dp(12f))
            setColorFilter(Color.WHITE)
            layoutParams = FrameLayout.LayoutParams(dp(48f), dp(48f))
        }

        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            val bg = GradientDrawable().apply { cornerRadius = dp(14f).toFloat(); setColor(0xF0FFFFFF.toInt()); setStroke(1, 0x20000000) }
            background = bg; setPadding(dp(12f), dp(10f), dp(12f), dp(10f))
            elevation = dp(6f).toFloat()
            layoutParams = FrameLayout.LayoutParams(dp(240f), FrameLayout.LayoutParams.WRAP_CONTENT)
        }

        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val tvTitle = TextView(this).apply {
            text = "悬浮翻译"; textSize = 14f; setTypeface(null, Typeface.BOLD)
            setTextColor(0xFF333333.toInt())
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        btnMic = ImageView(this).apply {
            setImageResource(android.R.drawable.ic_btn_speak_now)
            setPadding(dp(6f), dp(6f), dp(6f), dp(6f))
            val bg2 = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(0xFF5B5FC7.toInt()) }
            background = bg2; setColorFilter(Color.WHITE)
            layoutParams = LinearLayout.LayoutParams(dp(32f), dp(32f)).apply { marginStart = dp(6f) }
            setOnClickListener {
                if (mediaCaptureSyncMode) {
                    // Toggle translate on/off for media capture
                    mediaCaptureTranslateEnabled = !mediaCaptureTranslateEnabled
                    updateMicIcon(mediaCaptureTranslateEnabled)
                    updatePartial(if (mediaCaptureTranslateEnabled) "媒体翻译中…" else "媒体捕获中 · 点击麦克风开始翻译")
                    return@setOnClickListener
                }
                if (recording) stopRecording() else startRecording()
            }
        }
        val btnCollapse = ImageView(this).apply {
            setImageResource(android.R.drawable.ic_menu_close_clear_cancel)
            setPadding(dp(6f), dp(6f), dp(6f), dp(6f))
            layoutParams = LinearLayout.LayoutParams(dp(28f), dp(28f)).apply { marginStart = dp(4f) }
            setOnClickListener { toggleExpand(root, bubble, card) }
        }
        val btnClose = ImageView(this).apply {
            setImageResource(android.R.drawable.ic_delete)
            setPadding(dp(6f), dp(6f), dp(6f), dp(6f))
            layoutParams = LinearLayout.LayoutParams(dp(28f), dp(28f)).apply { marginStart = dp(4f) }
            setOnClickListener { stopSelf() }
        }
        header.addView(tvTitle); header.addView(btnMic); header.addView(btnCollapse); header.addView(btnClose)

        tvPartial = TextView(this).apply {
            textSize = 12f; setTextColor(0xFF888888.toInt())
            maxLines = 1; setPadding(0, dp(4f), 0, 0)
        }
        tvTranslation = TextView(this).apply {
            textSize = 13f; setTextColor(0xFF222222.toInt())
            maxLines = 4; setPadding(0, dp(4f), 0, 0)
        }

        card.addView(header); card.addView(tvPartial); card.addView(tvTranslation)
        containerCard = card

        root.addView(card); root.addView(bubble)
        bubble.visibility = View.GONE

        bubble.setOnClickListener { toggleExpand(root, bubble, card) }

        floatingView = root

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            if (Build.VERSION.SDK_INT >= 26) WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            else @Suppress("DEPRECATION") WindowManager.LayoutParams.TYPE_PHONE,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = dp(20f); y = dp(100f)
        }

        root.setOnTouchListener(object : View.OnTouchListener {
            var ix = 0; var iy = 0; var px = 0f; var py = 0f; var moved = false
            override fun onTouch(v: View, e: MotionEvent): Boolean {
                when (e.action) {
                    MotionEvent.ACTION_DOWN -> { ix = params.x; iy = params.y; px = e.rawX; py = e.rawY; moved = false }
                    MotionEvent.ACTION_MOVE -> {
                        params.x = ix + (e.rawX - px).toInt(); params.y = iy + (e.rawY - py).toInt()
                        windowManager?.updateViewLayout(floatingView, params); moved = true
                    }
                }
                return moved
            }
        })

        windowManager?.addView(root, params)
    }

    private fun toggleExpand(root: FrameLayout, bubble: ImageView, card: View) {
        expanded = !expanded
        if (expanded) { card.visibility = View.VISIBLE; bubble.visibility = View.GONE }
        else { card.visibility = View.GONE; bubble.visibility = View.VISIBLE }
    }

    private fun updatePartial(text: String) { tvPartial?.post { tvPartial?.text = text } }
    private fun updateTranslation(text: String) { tvTranslation?.post { tvTranslation?.text = text } }
    private fun updateMicIcon(rec: Boolean) {
        btnMic?.post {
            if (mediaCaptureSyncMode) {
                // Green = translating, blue = just capturing (waiting for user)
                val color = if (mediaCaptureTranslateEnabled) 0xFF4CAF50.toInt() else 0xFF5B5FC7.toInt()
                val bg = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(color) }
                btnMic?.background = bg
                btnMic?.setImageResource(if (mediaCaptureTranslateEnabled) android.R.drawable.ic_media_play else android.R.drawable.ic_btn_speak_now)
                btnMic?.isEnabled = true // user can toggle translate
            } else {
                val bg = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(if (rec) 0xFFEF5350.toInt() else 0xFF5B5FC7.toInt()) }
                btnMic?.background = bg
                btnMic?.setImageResource(android.R.drawable.ic_btn_speak_now)
                btnMic?.isEnabled = true
            }
        }
    }

    private fun copyAssetDir(a: String, d: File) {
        if (!d.exists()) d.mkdirs()
        for (n in assets.list(a) ?: emptyArray()) {
            val p = "$a/$n"; val c = assets.list(p)
            if (c.isNullOrEmpty()) assets.open(p).use { i -> FileOutputStream(File(d, n)).use { o -> i.copyTo(o) } }
            else copyAssetDir(p, File(d, n))
        }
    }
}

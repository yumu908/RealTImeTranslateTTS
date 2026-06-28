package com.example.myapplication1

import android.Manifest
import android.annotation.SuppressLint
import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.*
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.BatteryManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import com.example.myapplication1.asr.LanguageDetector
import com.example.myapplication1.asr.SpeechEnhancer
import com.example.myapplication1.asr.SherpaStreamingAsr
import com.example.myapplication1.asr.SherpaWhisperAsr
import com.example.myapplication1.asr.WhisperApiAsr
import com.example.myapplication1.tts.EdgeTts
import com.example.myapplication1.tts.KokoroTts
import com.example.myapplication1.tts.VitsTts
import com.example.myapplication1.translation.*
import com.example.myapplication1.translation.OnDeviceTranslationModel
import com.example.myapplication1.translation.TranslationModelManager
import com.example.myapplication1.tts.GoogleTranslateTts
import com.example.myapplication1.tts.OpenAiTts
import com.example.myapplication1.ui.theme.MyApplication1Theme
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.Translation
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.sync.Semaphore
import com.example.myapplication1.translation.LatencyMode
import com.example.myapplication1.translation.TranslationContext
import com.example.myapplication1.translation.TranslationMeta
import com.example.myapplication1.translation.GlossaryManager
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.log10
import kotlin.math.max

class MainActivity : ComponentActivity() {

    // ---- 段落模型 ----
    data class Segment(
        val seqId: Int = -1,
        val en: String,
        val zh: String = "",
        val translating: Boolean = true,
        /** Quality-upgraded translation (SWR). Empty if no upgrade available. */
        val qualityZh: String = "",
        /** Route that produced the current best translation: "fast", "quality", "fallback". */
        val route: String = "fast"
    )
    data class Paragraph(
        val id: Int,
        val segments: List<Segment> = emptyList(),
        /** 从历史加载的只读段落（不可再追加新句子）。 */
        val fromHistory: Boolean = false
    ) {
        val combinedEn: String get() = segments.joinToString(" ") { it.en }
        val rawZh: String get() = segments.filter { it.zh.isNotBlank() }.joinToString("") {
            it.qualityZh.ifBlank { it.zh }
        }
        val anyTranslating: Boolean get() = segments.any { it.translating }
        val allDone: Boolean get() = segments.isNotEmpty() && segments.none { it.translating }
    }

    // Using mutableStateOf(List) instead of mutableStateListOf because
    // SnapshotStateList element replacement does NOT reliably trigger
    // LazyColumn item recomposition. With mutableStateOf, every write
    // creates a new List reference that Compose always detects.
    private var _paragraphs by mutableStateOf(listOf<Paragraph>())

    // Helpers for immutable list mutations (each creates a new List reference)
    private fun paragraphsAdd(p: Paragraph) { _paragraphs = _paragraphs + p }
    private fun paragraphsSet(i: Int, p: Paragraph) { _paragraphs = _paragraphs.toMutableList().also { it[i] = p } }
    private fun paragraphsClear() { _paragraphs = emptyList() }
    private var _nextParagraphId = 0
    private var _currentPartial by mutableStateOf("")
    private var _recording by mutableStateOf(false)
    private var _micHeardVoice by mutableStateOf(false)

    // ---- 设置 ----
    private var _uiLang by mutableStateOf("zh")  // "zh" or "en"
    private var _autoSpeak by mutableStateOf(true)
    private var _enableTranslation by mutableStateOf(true)
    // ASR: 0=系统, 1=Vosk, 2=OpenAI, 3=Groq, 4=本地Whisper
    private var _asrEngine by mutableStateOf(0)
    // TTS: 0=Edge, 1=系统, 2=Google, 3=OpenAI
    private var _ttsEngine by mutableStateOf(0)
    private var _edgeVoiceIdx by mutableStateOf(0)
    private var _openaiVoiceIdx by mutableStateOf(4) // nova
    private var _openaiKey by mutableStateOf("")
    private var _groqKey by mutableStateOf("")
    private var _whisperModelIdx by mutableStateOf(0)

    // Claude API (Anthropic)
    private var _claudeKey by mutableStateOf("")
    private var _claudeTransModel by mutableStateOf("claude-sonnet-4-20250514")

    // 火山引擎 TTS（字节跳动）
    private var _volcanoAppId by mutableStateOf("")
    private var _volcanoToken by mutableStateOf("")
    private var _volcanoCluster by mutableStateOf(com.example.myapplication1.tts.VolcanoTts.DEFAULT_CLUSTER)
    private var _volcanoVoiceIdx by mutableStateOf(0)

    // 翻译引擎: 0=MLKit, 1=OpenAI, 2=Groq, 3=DeepL, 4=本地服务器, 5=Opus-MT, 6=NLLB
    private var _translationEngineType by mutableStateOf(0)
    private var _deeplKey by mutableStateOf("")
    private var _localServerUrl by mutableStateOf("http://192.168.1.100:11434/v1")
    private var _localServerModel by mutableStateOf("qwen2.5:7b")

    // AI 润色: 用快速 LLM API 修正离线翻译结果
    // 0=关闭, 1=Groq, 2=OpenAI, 3=本地服务器, 4=手机本机
    private var _refineProvider by mutableStateOf(0)
    private var _refineModel by mutableStateOf("llama-3.3-70b-versatile")
    private var _refineServerUrl by mutableStateOf("http://192.168.1.100:11434/v1")

    // ---- 翻译上下文增强 ----
    // 0=实时(REALTIME), 1=平衡(BALANCED), 2=质量(QUALITY)
    private var _latencyMode by mutableStateOf(0)
    private var _backgroundText by mutableStateOf("")
    // "auto", "general", "meeting", "medical", "customer_support", "game"
    private var _domainHint by mutableStateOf("auto")
    private var _showBackgroundSheet by mutableStateOf(false)

    // ---- 音频设备 ----
    private var _inputDevices = mutableStateListOf<AudioDeviceInfo>()
    private var _outputDevices = mutableStateListOf<AudioDeviceInfo>()
    private var _selectedInputId by mutableStateOf(0)   // 0 = default
    private var _selectedOutputId by mutableStateOf(0)

    // ---- 媒体转译 ----
    private var _mediaCaptureActive by mutableStateOf(false)
    private var _mediaCaptureStatus by mutableStateOf("")
    private var _mediaSourceApp by mutableStateOf("")
    private var _mediaCaptureError by mutableStateOf("")

    // ---- 段落管理 ----
    /** Silence-based paragraph break: fire after this many ms without a new ASR result. */
    private val PARAGRAPH_SILENCE_MS = 4000L
    private var paragraphSilenceJob: Job? = null

    // ---- 翻译引擎缓存 ----
    private var cachedTransEngine: TranslationEngine? = null
    private var cachedTransEngineType = -1

    // ---- 翻译历史 ----
    private lateinit var translationHistory: TranslationHistory
    private var _showHistory by mutableStateOf(false)

    // ---- TTS 语速自适应 ----
    private val ttsPendingCount = AtomicInteger(0)

    // ---- 翻译流水线 ----
    private lateinit var translationPipeline: TranslationPipeline

    // ---- 延迟指标 ----
    private var _asrLatencyMs by mutableStateOf(0L)
    private var _transLatencyMs by mutableStateOf(0L)
    private var _refineLatencyMs by mutableStateOf(0L)
    private var _ttsLatencyMs by mutableStateOf(0L)
    private var _ttsQueueSize by mutableStateOf(0)
    private var _ttsSpeedPct by mutableStateOf(0)

    // ---- TTS 回声抑制 ----
    @Volatile private var _isTtsSpeaking = false
    private var _ttsSpeakEndTime = 0L
    /** Minimum grace period after TTS ends before ASR results are accepted again. */
    private val TTS_ECHO_GRACE_BASE_MS = 200L
    /** Extra ms per Chinese character in last TTS output (speech lingers after audio). */
    private val TTS_ECHO_GRACE_PER_CHAR_MS = 15L
    @Volatile private var _lastTtsLength = 0

    /** Dynamic echo grace: longer TTS output = longer grace period to catch reverb. */
    private fun ttsEchoGraceMs(): Long =
        TTS_ECHO_GRACE_BASE_MS + (_lastTtsLength * TTS_ECHO_GRACE_PER_CHAR_MS).coerceAtMost(1500L)

    // ---- 翻译缓存命中指标 ----
    private var _transCacheHits by mutableStateOf(0L)

    // ---- 设备指标 ----
    private var _cpuUsagePct by mutableStateOf(0f)
    private var _memoryUsageMB by mutableStateOf(0)
    private var _batteryPct by mutableStateOf(0)
    private var _batteryTempC by mutableStateOf(0f)
    private var deviceMetricsJob: Job? = null
    private var lastCpuTime = 0L
    private var lastWallTime = 0L

    // ---- 会话管理 ----
    private var _showSessionDialog by mutableStateOf(false)
    private var _historySessions = mutableStateListOf<TranslationHistory.Session>()
    private var _historySearchQuery by mutableStateOf("")
    private var _editingSessionId by mutableStateOf<String?>(null)
    private var _editingTitle by mutableStateOf("")
    private var _autoFloatingOnPause by mutableStateOf(false)
    // ---- 离线翻译模型 ----
    private var _offlineTransModelIdx by mutableStateOf(0)
    private var _offlineTransDownloading by mutableStateOf(false)
    private var _offlineTransProgress by mutableStateOf("")
    private lateinit var translationModelManager: TranslationModelManager

    private var _smartFilterEnabled by mutableStateOf(true)
    private var _filterFillers by mutableStateOf(true)
    private var _filterEcho by mutableStateOf(true)
    private var _filterNoise by mutableStateOf(true)
    private var _filterMusic by mutableStateOf(true)

    // ---- Iter-3: AI 降噪 ----
    private var _denoiserEnabled by mutableStateOf(false)
    private var _denoiserReady by mutableStateOf(false)
    private var _denoiserDownloading by mutableStateOf(false)
    private var _denoiserProgress by mutableStateOf("")
    private lateinit var speechEnhancer: SpeechEnhancer

    // ---- Iter-4: 语种检测 & 双向互译 ----
    private var _sourceLang by mutableStateOf("en")
    private var _targetLang by mutableStateOf("zh")
    private var _langAutoMode by mutableStateOf(true)
    private var _detectedLang by mutableStateOf("")
    private var _lidReady by mutableStateOf(false)
    private var _lidDownloading by mutableStateOf(false)
    private var _lidProgress by mutableStateOf("")
    private lateinit var languageDetector: LanguageDetector

    // ---- API 测试 ----
    private val apiTestManager = ApiTestManager()
    private var _apiTestResults = mutableStateListOf<ApiTestManager.ApiTestResult>()
    private var _apiTestRunning by mutableStateOf(false)
    private var _apiTestProgress by mutableStateOf("")
    // API Key 管理
    private var _showApiKeyManager by mutableStateOf(false)
    // 术语库管理
    private var _glossaryImportDomain by mutableStateOf("general")
    private var _glossaryDownloading by mutableStateOf(false)
    private var _glossaryImportResult by mutableStateOf("")
    private var _newKeyName by mutableStateOf("")
    private var _newKeyValue by mutableStateOf("")

    // ---- ONNX Runtime 执行提供器 (GPU/NPU 加速) ----
    // "cpu" | "nnapi" | "xnnpack" — 统一控制所有 Sherpa ASR/TTS 与 OnDeviceTranslation
    private var _ortProvider by mutableStateOf(AccelerationConfig.CPU)

    // ---- 设置页导航（三级菜单） ----
    private enum class SettingsScreen { Main, General, Voice, Translation, Advanced }
    private var _settingsScreen by mutableStateOf(SettingsScreen.Main)

    // ---- 段落聚合配置 ----
    private var _paragraphAutoGroup by mutableStateOf(true)
    private var _paragraphMaxSegments by mutableStateOf(8)
    private var _paragraphSilenceMs by mutableStateOf(4000)

    // ---- 会话删除确认 ----
    private var _pendingDeleteSessionId by mutableStateOf<String?>(null)
    private var _showClearAllDialog by mutableStateOf(false)

    // ---- 状态 ----
    private var _voskReady by mutableStateOf(false)
    private var _translatorReady by mutableStateOf(false)
    private var _ttsReady by mutableStateOf(false)
    private var _ttsLangOk by mutableStateOf(false)
    private var _systemAsrAvailable by mutableStateOf(false)
    private var _localWhisperReady by mutableStateOf(false)
    private var _localWhisperDownloading by mutableStateOf(false)
    private var _downloadProgress by mutableStateOf("")

    // ---- Streaming ASR (Iter-1) ----
    private var _streamingAsrModelIdx by mutableStateOf(0)
    private var _streamingAsrReady by mutableStateOf(false)
    private var _streamingAsrDownloading by mutableStateOf(false)
    private var _streamingAsrProgress by mutableStateOf("")
    private lateinit var streamingAsr: SherpaStreamingAsr

    // ---- Kokoro TTS (English) ----
    private var _kokoroVoiceSid by mutableStateOf(0)
    private var _kokoroReady by mutableStateOf(false)
    private var _kokoroDownloading by mutableStateOf(false)
    private var _kokoroProgress by mutableStateOf("")
    private lateinit var kokoroTts: KokoroTts

    // ---- Chinese TTS (VITS/Matcha) ----
    private lateinit var vitsTts: VitsTts
    private var _vitsTtsReady by mutableStateOf(false)
    private var _zhTtsModelIdx by mutableStateOf(0)
    private var _zhTtsDownloading by mutableStateOf(false)
    private var _zhTtsProgress by mutableStateOf("")
    private var _logs = mutableStateListOf<String>()

    // ---- Vosk ----
    private var audioRecord: AudioRecord? = null
    private var recognizer: Recognizer? = null
    private var voskModel: Model? = null
    private var asrJob: Job? = null
    private var aec: android.media.audiofx.AcousticEchoCanceler? = null

    // ---- System ASR ----
    private var speechRecognizer: SpeechRecognizer? = null
    private var systemAsrContinue = false

    // ---- Whisper ----
    private var whisperAsr: WhisperApiAsr? = null
    private lateinit var sherpaWhisperAsr: SherpaWhisperAsr

    // ---- TTS ----
    private var systemTts: TextToSpeech? = null
    private lateinit var edgeTts: EdgeTts
    private val googleTts = GoogleTranslateTts()
    private var openAiTts: OpenAiTts? = null
    private var volcanoTts: com.example.myapplication1.tts.VolcanoTts? = null
    private var volcanoTtsCreds: String = ""  // cached "appId|token|cluster" for invalidation
    private val ttsQueue = Channel<String>(Channel.UNLIMITED)
    private var ttsConsumerJob: Job? = null

    // ---- 翻译 ----
    private val translator by lazy {
        Translation.getClient(
            com.google.mlkit.nl.translate.TranslatorOptions.Builder()
                .setSourceLanguage(TranslateLanguage.ENGLISH)
                .setTargetLanguage(TranslateLanguage.CHINESE).build()
        )
    }

    private val glossaryFilePicker = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            val fileName = uri.lastPathSegment?.substringAfterLast('/') ?: "glossary.csv"
            val result = GlossaryManager.importUserFile(this, uri, _glossaryImportDomain, fileName)
            _glossaryImportResult = if (result.success) "已导入 ${result.entryCount} 条术语" else result.error
        }
    }

    private val requestAudioPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { if (it) { log(S("mic_permission")); prepareAll() } else log("未授予麦克风权限") }

    private val requestMediaProjection = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK && result.data != null) {
            // Pass result to MediaCaptureService (foreground service already started)
            val svcIntent = Intent(this, MediaCaptureService::class.java).apply {
                action = MediaCaptureService.ACTION_START
                putExtra(MediaCaptureService.EXTRA_RESULT_CODE, result.resultCode)
                putExtra(MediaCaptureService.EXTRA_RESULT_DATA, result.data)
            }
            startService(svcIntent)
            _mediaCaptureActive = true
            _mediaCaptureStatus = "启动中…"
        } else {
            log("用户拒绝了媒体投影权限")
            _mediaCaptureActive = false
            try { stopService(Intent(this, MediaCaptureService::class.java)) } catch (_: Throwable) {}
        }
    }

    // ===================== Lifecycle =====================

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        edgeTts = EdgeTts(cacheDir)
        sherpaWhisperAsr = SherpaWhisperAsr(this)
        streamingAsr = SherpaStreamingAsr(this)
        kokoroTts = KokoroTts(this)
        vitsTts = VitsTts(this)
        speechEnhancer = SpeechEnhancer(this)
        languageDetector = LanguageDetector(this)
        translationModelManager = TranslationModelManager(this)
        translationHistory = TranslationHistory(filesDir)
        _historySessions.addAll(translationHistory.load())
        loadSettings()
        refreshAudioDevices()
        // Initialize glossary system
        GlossaryManager.init(this)
        // Initialize translation pipeline with ordered delivery
        translationPipeline = TranslationPipeline(lifecycleScope)
        translationPipeline.enableTranslation = _enableTranslation
        translationPipeline.setCallback(pipelineCallback)
        translationPipeline.setEngine(getTranslationEngine())
        translationPipeline.setRefiner(buildRefiner())
        updatePipelineContext()
        // Show session dialog if there are previous sessions
        if (_historySessions.isNotEmpty()) _showSessionDialog = true
        else translationHistory.newSession()
        FloatingTranslateService.translationCallback = floatingTranslationCb
        MediaCaptureService.asrCallback = mediaCaptureAsrCb
        setContent { MyApplication1Theme { AppUI() } }
        requestAudioPermission.launch(Manifest.permission.RECORD_AUDIO)
        startTtsConsumer()
    }

    override fun onPause() {
        super.onPause()
        // Auto-launch floating window when leaving app if recording or media capture active
        if ((_recording || _mediaCaptureActive) && !FloatingTranslateService.isRunning) {
            if (Settings.canDrawOverlays(this)) {
                FloatingTranslateService.autoLaunched = true
                FloatingTranslateService.appWasRecording = _recording
                val svcIntent = Intent(this, FloatingTranslateService::class.java)
                if (Build.VERSION.SDK_INT >= 26) startForegroundService(svcIntent) else startService(svcIntent)
                _autoFloatingOnPause = true
            }
        }
    }

    override fun onResume() {
        super.onResume()
        // Stop auto-launched floating window when returning
        if (_autoFloatingOnPause && FloatingTranslateService.autoLaunched) {
            try { stopService(Intent(this, FloatingTranslateService::class.java)) } catch (_: Throwable) {}
            FloatingTranslateService.autoLaunched = false
            FloatingTranslateService.appWasRecording = false
            _autoFloatingOnPause = false
        }
        // Reload history to catch floating window additions
        _historySessions.clear()
        _historySessions.addAll(translationHistory.load())
        // Re-register callbacks
        FloatingTranslateService.translationCallback = floatingTranslationCb
        // Only set asrCallback if floating service hasn't chained it
        if (!FloatingTranslateService.isRunning) {
            MediaCaptureService.asrCallback = mediaCaptureAsrCb
        }
        // Sync media capture state from service
        _mediaCaptureActive = MediaCaptureService.isCapturing
        if (_mediaCaptureActive) {
            _mediaCaptureStatus = MediaCaptureService.captureStatus
            _mediaSourceApp = MediaCaptureService.sourceAppName
        }
    }

    private val floatingTranslationCb = object : FloatingTranslateService.Companion.TranslationCallback {
        override fun onFloatingTranslation(en: String, zh: String) {
            runOnUiThread {
                // Add floating window translation to current paragraph
                if (_paragraphs.isEmpty()) paragraphsAdd(Paragraph(id = _nextParagraphId++))
                val pi = _paragraphs.lastIndex
                val p = _paragraphs[pi]
                paragraphsSet(pi, p.copy(segments = p.segments + Segment(en = en, zh = zh, translating = false)))
                _historySessions.clear()
                _historySessions.addAll(translationHistory.allSessions())
            }
        }
        override fun onRecordingStateChanged(isRecording: Boolean) {
            // Sync floating window recording state (informational)
            runOnUiThread {
                log(if (isRecording) "悬浮窗开始录音" else "悬浮窗停止录音")
            }
        }
    }

    private val mediaCaptureAsrCb = object : MediaCaptureService.Companion.AsrCallback {
        override fun onPartial(text: String) {
            _currentPartial = text
        }
        override fun onResult(text: String) { onAsrResult(text) }
        override fun onStateChanged(capturing: Boolean, status: String, sourceApp: String) {
            _mediaCaptureActive = capturing
            _mediaCaptureStatus = status
            _mediaSourceApp = sourceApp
            if (!capturing && status.startsWith("错误")) {
                _mediaCaptureError = status
            }
        }
        override fun onError(msg: String) {
            _mediaCaptureError = msg
            log("媒体捕获: $msg")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        FloatingTranslateService.translationCallback = null
        MediaCaptureService.asrCallback = null
        paragraphSilenceJob?.cancel(); paragraphSilenceJob = null
        ttsConsumerJob?.cancel(); ttsQueue.close(); stopDeviceMetrics()
        if (::translationPipeline.isInitialized) translationPipeline.close()
        stopAllAsr(); stopMediaCapture()
        try { speechRecognizer?.destroy() } catch (_: Throwable) {}
        try { recognizer?.close() } catch (_: Throwable) {}
        try { voskModel?.close() } catch (_: Throwable) {}
        try { translator.close() } catch (_: Throwable) {}
        try { systemTts?.stop(); systemTts?.shutdown() } catch (_: Throwable) {}
        sherpaWhisperAsr.release(); edgeTts.close(); googleTts.close()
        try { openAiTts?.close() } catch (_: Throwable) {}
        try { volcanoTts?.close() } catch (_: Throwable) {}
        if (::streamingAsr.isInitialized) streamingAsr.release()
        if (::kokoroTts.isInitialized) kokoroTts.release()
        if (::vitsTts.isInitialized) vitsTts.release()
        if (::speechEnhancer.isInitialized) speechEnhancer.release()
        if (::languageDetector.isInitialized) languageDetector.release()
        cachedTransEngine?.close(); translationHistory.close()
    }

    // ===================== Init =====================

    private fun prepareAll() {
        translator.downloadModelIfNeeded(DownloadConditions.Builder().requireWifi().build())
            .addOnSuccessListener { _translatorReady = true; log("翻译模型就绪") }
            .addOnFailureListener { log("翻译模型下载失败: ${it.message}") }

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val dst = File(filesDir, "model-en")
                if (!dst.exists()) copyAssetDir("model-en", dst)
                voskModel = Model(dst.absolutePath)
                withContext(Dispatchers.Main) { _voskReady = true; log("Vosk 模型就绪") }
            } catch (e: Throwable) { withContext(Dispatchers.Main) { log("Vosk: ${e.message}") } }
        }

        _systemAsrAvailable = SpeechRecognizer.isRecognitionAvailable(this)
        log(if (_systemAsrAvailable) "系统ASR可用" else "System ASR unavailable")

        systemTts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val r = systemTts?.setLanguage(Locale.SIMPLIFIED_CHINESE)
                _ttsLangOk = r != TextToSpeech.LANG_MISSING_DATA && r != TextToSpeech.LANG_NOT_SUPPORTED
                _ttsReady = true
                // 使用 USAGE_ASSISTANT 避免被 MediaCapture 捕获
                systemTts?.setAudioAttributes(AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANT)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build())
                log(if (_ttsLangOk) "系统TTS就绪" else "系统TTS不支持中文")
            } else { _ttsReady = false; log("系统TTS初始化失败") }
        }
        log("Edge TTS 就绪")

        // All sherpa-onnx native inits MUST be serialized — the native library
        // has global state that crashes on concurrent initialization (SIGABRT
        // "pthread_mutex_lock called on a destroyed mutex").
        lifecycleScope.launch(Dispatchers.IO) {
            val selModel = selectedWhisperModel()
            if (sherpaWhisperAsr.isModelDownloaded(selModel)) {
                val ok = sherpaWhisperAsr.initModel(selModel)
                withContext(Dispatchers.Main) {
                    _localWhisperReady = ok
                    log(if (ok) "本地Whisper [${selModel.label}] 就绪" else "本地Whisper失败")
                }
            }

            val selStreamModel = selectedStreamingModel()
            if (streamingAsr.isModelDownloaded(selStreamModel)) {
                val ok = streamingAsr.initModel(selStreamModel)
                withContext(Dispatchers.Main) {
                    _streamingAsrReady = ok
                    log(if (ok) "流式ASR [${selStreamModel.label}] 就绪" else "流式ASR初始化失败")
                }
            }

            // Iter-3: init denoiser
            if (speechEnhancer.isModelDownloaded()) {
                val ok = speechEnhancer.init()
                withContext(Dispatchers.Main) {
                    _denoiserReady = ok
                    if (ok) { streamingAsr.speechEnhancer = speechEnhancer; streamingAsr.denoiserEnabled = _denoiserEnabled }
                    log(if (ok) "AI降噪就绪" else "AI降噪初始化失败")
                }
            }

            // Iter-4: init language detector
            if (languageDetector.isModelDownloaded()) {
                val ok = languageDetector.init()
                withContext(Dispatchers.Main) {
                    _lidReady = ok
                    if (ok) { streamingAsr.languageDetector = languageDetector; streamingAsr.languageDetectionEnabled = _langAutoMode }
                    log(if (ok) "语种识别就绪" else "语种识别初始化失败")
                }
            }

            // Kokoro is NOT initialized here — user disabled it for TTS.
            // Only ONE OfflineTts instance can exist at a time (ONNX Runtime global mutex).

            // Clean up removed model directories
            try { File(filesDir, "sherpa-matcha-zh-en").deleteRecursively() } catch (_: Throwable) {}

            // Sherpa TTS: init selected model (fallback to AISHELL3 if unavailable)
            val zhModel = selectedZhTtsModel()
            val modelToInit = if (vitsTts.isModelDownloaded(zhModel)) zhModel else VitsTts.Model.AISHELL3
            val vitsOk = vitsTts.initModel(modelToInit)
            withContext(Dispatchers.Main) {
                _vitsTtsReady = vitsOk
                log(if (vitsOk) "${modelToInit.label} 就绪" else "${modelToInit.label} 初始化失败")
            }
        }
    }

    // ===================== Audio Devices =====================

    private fun refreshAudioDevices() {
        val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        _inputDevices.clear()
        _inputDevices.addAll(am.getDevices(AudioManager.GET_DEVICES_INPUTS))
        _outputDevices.clear()
        _outputDevices.addAll(am.getDevices(AudioManager.GET_DEVICES_OUTPUTS))
        // Preserve system default microphone for ASR (do not auto-select media audio here)
        // If a specific input device is required, the UI can let the user choose it.
        // The previous auto‑selection logic has been removed to avoid quality degradation.
        // if (_selectedInputId == 0 && _inputDevices.isNotEmpty()) {
        //     // Try to find a device whose name indicates it is the media playback capture
        //     val mediaDevice = _inputDevices.firstOrNull {
        //         val name = it.productName?.toString() ?: ""
        //         name.contains("媒体") || name.contains("Media")
        //     }
        //     // Fallback: choose the last input device that is not a built‑in microphone
        //     val fallback = _inputDevices.filter { it.type != AudioDeviceInfo.TYPE_BUILTIN_MIC }.lastOrNull()
        //     val chosen = mediaDevice ?: fallback ?: _inputDevices.last()
        //     _selectedInputId = chosen.id
        //     saveInt("selected_input_id", chosen.id)
        // }
    }

    private fun deviceLabel(d: AudioDeviceInfo): String {
        val name = d.productName?.toString()?.takeIf { it.isNotBlank() } ?: ""
        val type = when (d.type) {
            AudioDeviceInfo.TYPE_BUILTIN_MIC -> "内置麦克风"
            AudioDeviceInfo.TYPE_BUILTIN_SPEAKER -> "内置扬声器"
            AudioDeviceInfo.TYPE_WIRED_HEADSET -> "有线耳机"
            AudioDeviceInfo.TYPE_WIRED_HEADPHONES -> "有线耳机"
            AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> "蓝牙SCO"
            AudioDeviceInfo.TYPE_BLUETOOTH_A2DP -> "蓝牙A2DP"
            AudioDeviceInfo.TYPE_USB_DEVICE -> "USB设备"
            AudioDeviceInfo.TYPE_USB_HEADSET -> "USB耳机"
            AudioDeviceInfo.TYPE_BUILTIN_EARPIECE -> "听筒"
            AudioDeviceInfo.TYPE_TELEPHONY -> "电话"
            else -> "设备${d.type}"
        }
        return if (name.isNotBlank() && !type.contains(name)) "$type ($name)" else type
    }

    private fun findInputDevice(): AudioDeviceInfo? {
        if (_selectedInputId == 0) return null
        return _inputDevices.firstOrNull { it.id == _selectedInputId }
    }

    private fun findOutputDevice(): AudioDeviceInfo? {
        if (_selectedOutputId == 0) return null
        return _outputDevices.firstOrNull { it.id == _selectedOutputId }
    }

    // ===================== TTS =====================

    /**
     * Maximum TTS queue depth before we start dropping intermediate items.
     * At this depth we drain all but the newest item in the queue so playback
     * catches up to the live translation stream (duplex timeliness).
     */
    private val TTS_QUEUE_DRAIN_THRESHOLD = 4

    private fun startTtsConsumer() {
        ttsConsumerJob = lifecycleScope.launch {
            while (true) {
                val text = ttsQueue.receiveCatching().getOrNull() ?: break
                _ttsQueueSize = ttsPendingCount.get()

                // Smart queue drain: if too many items queued, skip intermediate ones
                // to catch up to the live speech stream (improves duplex timeliness).
                var textToSpeak = text
                if (ttsPendingCount.get() > TTS_QUEUE_DRAIN_THRESHOLD) {
                    var skipped = 0
                    while (ttsPendingCount.get() > 1) {
                        val next = ttsQueue.tryReceive().getOrNull() ?: break
                        ttsPendingCount.decrementAndGet()
                        skipped++
                        textToSpeak = next  // keep moving forward to latest item
                    }
                    if (skipped > 0) Log.d("VRI", "TTS queue drain: skipped $skipped items")
                    _ttsQueueSize = ttsPendingCount.get()
                }

                if (_autoSpeak) {
                    val ttsStart = System.currentTimeMillis()
                    try {
                        // Use the SAME speakZh path for all engines (auto + manual).
                        // This ensures consistent behavior and fallback on failure.
                        speakZh(textToSpeak)
                        _ttsLatencyMs = System.currentTimeMillis() - ttsStart
                    } catch (e: Throwable) {
                        Log.e("VRI", "TTS error: ${e.message}")
                        _isTtsSpeaking = false
                        _ttsSpeakEndTime = System.currentTimeMillis()
                    }
                }
                ttsPendingCount.decrementAndGet()
                _ttsQueueSize = ttsPendingCount.get()
            }
        }
    }

    private fun startDeviceMetrics() {
        deviceMetricsJob?.cancel()
        deviceMetricsJob = lifecycleScope.launch(Dispatchers.IO) {
            while (isActive) {
                // CPU usage from /proc/self/stat
                try {
                    val statLine = java.io.File("/proc/self/stat").readText().split(" ")
                    val utime = statLine[13].toLong()
                    val stime = statLine[14].toLong()
                    val cpuTime = utime + stime
                    val wallTime = android.os.SystemClock.elapsedRealtime()
                    if (lastWallTime > 0) {
                        val dWall = wallTime - lastWallTime
                        val dCpu = (cpuTime - lastCpuTime) * 10 // jiffies → ms (assuming HZ=100)
                        if (dWall > 0) _cpuUsagePct = (dCpu.toFloat() / dWall * 100f).coerceIn(0f, 100f * Runtime.getRuntime().availableProcessors())
                    }
                    lastCpuTime = cpuTime
                    lastWallTime = wallTime
                } catch (_: Throwable) {}

                // Memory
                val rt = Runtime.getRuntime()
                _memoryUsageMB = ((rt.totalMemory() - rt.freeMemory()) / (1024 * 1024)).toInt()

                // Battery
                try {
                    val bm = getSystemService(BATTERY_SERVICE) as android.os.BatteryManager
                    _batteryPct = bm.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY)
                    val intent = registerReceiver(null, android.content.IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED))
                    val temp = intent?.getIntExtra(android.os.BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
                    _batteryTempC = temp / 10f
                } catch (_: Throwable) {}

                delay(2000)
            }
        }
    }

    private fun stopDeviceMetrics() {
        deviceMetricsJob?.cancel()
        deviceMetricsJob = null
    }

    private fun dynamicSpeedPct(): Int {
        val p = ttsPendingCount.get()
        return when { p <= 1 -> 0; p == 2 -> 15; p == 3 -> 25; p == 4 -> 35; else -> 50 }
    }

    private fun dynamicEdgeRate(): String {
        val pct = dynamicSpeedPct()
        _ttsSpeedPct = pct
        return "+${pct}%"
    }

    private fun dynamicSystemRate(): Float {
        val pct = dynamicSpeedPct()
        _ttsSpeedPct = pct
        return 1.0f + pct / 100f
    }

    private fun dynamicOpenAiSpeed(): Float {
        val pct = dynamicSpeedPct()
        _ttsSpeedPct = pct
        return (1.0f + pct / 100f).coerceIn(0.25f, 4.0f)
    }

    private suspend fun speakZh(text: String) {
        _isTtsSpeaking = true
        _lastTtsLength = text.length
        try {
            when (_ttsEngine) { 0 -> speakEdge(text); 1 -> speakSystem(text); 2 -> speakGoogle(text); 3 -> speakOpenAi(text); 4 -> speakKokoro(text); 5 -> speakVolcano(text); else -> speakEdge(text) }
        } finally {
            _isTtsSpeaking = false
            _ttsSpeakEndTime = System.currentTimeMillis()
        }
    }

    private suspend fun speakEdge(text: String) {
        val voices = EdgeTts.voicesForLang(_targetLang)
        val voice = voices.getOrNull(_edgeVoiceIdx)?.first ?: voices.firstOrNull()?.first ?: "en-US-JennyNeural"
        try { edgeTts.speak(text, voice = voice, rate = dynamicEdgeRate()) }
        catch (_: Throwable) { speakSystem(text) }
    }

    private suspend fun speakSystem(text: String) {
        if (_ttsReady && _ttsLangOk) {
            val rate = dynamicSystemRate()
            systemTts?.setSpeechRate(rate)
            systemTts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "u-${System.currentTimeMillis()}")
            delay((text.length * 280L / rate).toLong().coerceIn(500, 12000)); return
        }
        speakGoogle(text)
    }

    private suspend fun speakGoogle(text: String) {
        val speed = dynamicSystemRate()
        val locale = when (_targetLang) {
            "zh" -> "zh-CN"; "en" -> "en-US"; "ja" -> "ja-JP"; "ko" -> "ko-KR"
            "fr" -> "fr-FR"; "de" -> "de-DE"; "es" -> "es-ES"; "ru" -> "ru-RU"
            else -> "en-US"
        }
        try { googleTts.speak(text, locale, speed) } catch (e: Throwable) { log("TTS: ${e.message}") }
    }

    private suspend fun speakOpenAi(text: String) {
        if (_openaiKey.isBlank()) { speakEdge(text); return }
        try {
            if (openAiTts == null) openAiTts = OpenAiTts(_openaiKey, cacheDir)
            val voice = OpenAiTts.VOICES.getOrNull(_openaiVoiceIdx)?.first ?: "nova"
            openAiTts?.speak(text, voice = voice, speed = dynamicOpenAiSpeed())
        } catch (e: Throwable) { log("OpenAI TTS: ${e.message}"); speakEdge(text) }
    }

    private suspend fun speakVolcano(text: String) {
        if (_volcanoAppId.isBlank() || _volcanoToken.isBlank()) {
            log("火山引擎 TTS 未配置 App ID / Token，回退至 Edge")
            speakEdge(text); return
        }
        try {
            val creds = "$_volcanoAppId|$_volcanoToken|$_volcanoCluster"
            if (volcanoTts == null || volcanoTtsCreds != creds) {
                volcanoTts?.close()
                volcanoTts = com.example.myapplication1.tts.VolcanoTts(
                    _volcanoAppId, _volcanoToken, _volcanoCluster, cacheDir
                )
                volcanoTtsCreds = creds
            }
            val voices = com.example.myapplication1.tts.VolcanoTts.voicesForLang(_targetLang)
            val voice = voices.getOrNull(_volcanoVoiceIdx)?.first ?: voices.first().first
            volcanoTts?.speak(text, voice = voice, speed = dynamicOpenAiSpeed())
        } catch (e: Throwable) { log("火山引擎 TTS: ${e.message}"); speakEdge(text) }
    }

    private fun speakManual(text: String) {
        if (_ttsEngine == 4) {
            if (::kokoroTts.isInitialized) kokoroTts.stopPlayback()
            if (::vitsTts.isInitialized) vitsTts.stopPlayback()
        }
        lifecycleScope.launch { speakZh(text) }
    }

    // ===================== ASR dispatch =====================

    private fun toggleRecording() {
        if (_mediaCaptureActive) {
            log("媒体捕获中，请先停止媒体捕获再使用麦克风")
            return
        }
        if (_recording) stopAllAsr() else startAsr()
    }

    private fun startAsr() {
        if (_recording) return
        _currentPartial = ""; _micHeardVoice = false
        startDeviceMetrics()
        when (_asrEngine) {
            0 -> startSystemAsr(); 1 -> startVoskAsr()
            2 -> startWhisperApi(WhisperApiAsr.Provider.OPENAI, _openaiKey)
            3 -> startWhisperApi(WhisperApiAsr.Provider.GROQ, _groqKey)
            4 -> startLocalWhisper()
            5 -> startWhisperApi(WhisperApiAsr.Provider.GPT4O_MINI, _openaiKey)
            6 -> startStreamingAsr()
        }
    }

    private fun stopAllAsr() {
        if (!_recording) return
        _recording = false
        log("停止录音")

        // Cancel any pending silence-based paragraph break timer
        paragraphSilenceJob?.cancel(); paragraphSilenceJob = null

        when (_asrEngine) {
            0 -> stopSystemAsr()
            1 -> { stopVosk(); flushVosk() }
            2, 3, 5 -> whisperAsr?.stopGracefully()
            4 -> sherpaWhisperAsr.stopGracefully()
            6 -> streamingAsr.stopGracefully()
        }

        _currentPartial = ""
        stopDeviceMetrics()
    }

    private fun flushVosk() {
        val r = recognizer?.finalResult?.let { JSONObject(it).optString("text") }.orEmpty().trim()
        if (r.isNotBlank()) onAsrResult(r)
        else if (_currentPartial.isNotBlank()) onAsrResult(_currentPartial)
    }

    // ===================== System ASR =====================

    private val systemAsrListener = object : RecognitionListener {
        override fun onReadyForSpeech(p: Bundle?) {}
        override fun onBeginningOfSpeech() { _micHeardVoice = true }
        override fun onRmsChanged(r: Float) {}
        override fun onBufferReceived(b: ByteArray?) {}
        override fun onEndOfSpeech() { _asrProcessStart = System.currentTimeMillis() }
        override fun onError(e: Int) {
            if (e == SpeechRecognizer.ERROR_NO_MATCH || e == SpeechRecognizer.ERROR_SPEECH_TIMEOUT) {
                if (systemAsrContinue && _recording) restartSystemAsr()
            } else if (systemAsrContinue && _recording) {
                lifecycleScope.launch { delay(500); if (_recording) restartSystemAsr() }
            }
        }
        override fun onResults(r: Bundle?) {
            if (_asrProcessStart > 0) _asrLatencyMs = System.currentTimeMillis() - _asrProcessStart
            val t = r?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.trim().orEmpty()
            if (t.isNotBlank()) onAsrResult(t)
            if (systemAsrContinue && _recording) restartSystemAsr()
        }
        override fun onPartialResults(p: Bundle?) {
            val t = p?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.trim().orEmpty()
            if (t.isNotBlank()) _currentPartial = t
        }
        override fun onEvent(t: Int, p: Bundle?) {}
    }

    private fun systemAsrIntent() = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 1500L)
        putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 1200L)
    }

    private fun startSystemAsr() {
        if (!_systemAsrAvailable) { log("System ASR unavailable"); _asrEngine = 1; startVoskAsr(); return }
        try {
            speechRecognizer?.destroy()
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this).apply {
                setRecognitionListener(systemAsrListener)
            }
            systemAsrContinue = true; _recording = true; log("开始录音 (系统)")
            speechRecognizer?.startListening(systemAsrIntent())
        } catch (e: Throwable) { log("系统ASR: ${e.message}"); _asrEngine = 1; startVoskAsr() }
    }

    private fun restartSystemAsr() {
        try { speechRecognizer?.startListening(systemAsrIntent()) } catch (_: Throwable) {}
    }

    private fun stopSystemAsr() { systemAsrContinue = false; try { speechRecognizer?.stopListening() } catch (_: Throwable) {} }

    // ===================== Vosk =====================

    private fun attachEchoCanceler(record: AudioRecord) {
        try {
            aec?.release(); aec = null
            if (android.media.audiofx.AcousticEchoCanceler.isAvailable()) {
                aec = android.media.audiofx.AcousticEchoCanceler.create(record.audioSessionId)
                aec?.enabled = true
                Log.d("MainActivity", "AEC enabled: ${aec?.enabled}")
            }
        } catch (e: Throwable) {
            Log.w("MainActivity", "AEC not available: ${e.message}")
        }
    }

    private fun releaseEchoCanceler() {
        aec?.release(); aec = null
    }

    @SuppressLint("MissingPermission")
    private fun startVoskAsr() {
        // 16 kHz 为模型固定采样率，使用更大的缓冲（32 KB）降低 CPU 轮询频率
        val sr = 16000
        val minBuf = AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val bs = max(minBuf, 32768) // 32 KB buffer
        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                sr,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bs
            )
        } catch (e: Throwable) {
            log("录音失败: ${e.message}")
            return
        }
        if (voskModel == null) { log("Vosk模型未就绪"); return }
        recognizer?.close(); recognizer = Recognizer(voskModel, sr.toFloat())
        findInputDevice()?.let { audioRecord?.preferredDevice = it }
        // 对于已关闭的智能过滤和翻译，回声消除往往不是必需，直接跳过以降低 CPU 开销
        // attachEchoCanceler(audioRecord!!)
        try { audioRecord?.startRecording() } catch (e: Throwable) { log("录音启动失败"); return }
        _recording = true; log("开始录音 (Vosk)")
        val localRec = recognizer ?: run { log("Vosk识别器未就绪"); return }
        val localAudio = audioRecord ?: return
        asrJob = lifecycleScope.launch(Dispatchers.IO) {
            val buf = ShortArray(2048)
            while (isActive && _recording) {
                val n = try { localAudio.read(buf, 0, buf.size) } catch (_: Throwable) { break }
                if (n <= 0) { if (n < 0) break else continue }
                // 简单 RMS 检测，仅用于激活 mic 图标，不做额外计算
                val rms = buf.take(n).fold(0.0) { a, s -> a + s * s } / n
                if (10 * kotlin.math.log10(rms + 1e-9) > -35.0) _micHeardVoice = true
                try {
                    if (localRec.acceptWaveForm(buf, n)) {
                        val t = JSONObject(localRec.result).optString("text").trim()
                        if (t.isNotBlank()) withContext(Dispatchers.Main) { onAsrResult(t) }
                    } else {
                        val p = JSONObject(localRec.partialResult).optString("partial").trim()
                        if (p.isNotBlank()) withContext(Dispatchers.Main) { _currentPartial = p }
                    }
                } catch (_: Throwable) { /* ignore */ }
            }
        }
    }

    private fun stopVosk() { asrJob?.cancel(); asrJob = null; releaseEchoCanceler(); try { audioRecord?.stop() } catch (_: Throwable) {}; try { audioRecord?.release() } catch (_: Throwable) {}; audioRecord = null }

    // ===================== Whisper API =====================

    private var _asrProcessStart = 0L
    private val whisperCb = object : WhisperApiAsr.Callback {
        override fun onListening() { _currentPartial = "" }
        override fun onSpeechDetected() { _micHeardVoice = true; _currentPartial = "语音检测中…" }
        override fun onProcessing() { _currentPartial = "正在识别…"; _asrProcessStart = System.currentTimeMillis() }
        override fun onResult(text: String) { if (_asrProcessStart > 0) _asrLatencyMs = System.currentTimeMillis() - _asrProcessStart; onAsrResult(text) }
        override fun onError(msg: String) { log("WhisperAPI: $msg") }
    }

    private fun startWhisperApi(provider: WhisperApiAsr.Provider, key: String) {
        if (key.isBlank()) { log("Set ${provider.name} API Key"); return }
        whisperAsr?.close(); whisperAsr = WhisperApiAsr(this, key, provider, language = _sourceLang)
        _recording = true

        // Start ASR IMMEDIATELY — don't wait for VAD download
        whisperAsr?.start(lifecycleScope, whisperCb)

        val vadAvailable = WhisperApiAsr.isVadAvailable(this)
        if (vadAvailable) {
            log("开始录音 (${provider.name} + Silero VAD)")
        } else {
            log("开始录音 (${provider.name} · 简易VAD · 正在下载Silero…)")
            // Download VAD in background — next recording session will use it
            lifecycleScope.launch(Dispatchers.IO) {
                val f = WhisperApiAsr.ensureVadDownloaded(this@MainActivity)
                withContext(Dispatchers.Main) {
                    if (f != null) log("Silero VAD 已下载，下次录音生效")
                    else log("Silero VAD 下载失败")
                }
            }
        }
    }

    // ===================== Local Whisper =====================

    private val localWhisperCb = object : SherpaWhisperAsr.Callback {
        override fun onListening() { _currentPartial = "" }
        override fun onSpeechDetected() { _micHeardVoice = true; _currentPartial = "语音检测中…" }
        override fun onProcessing() { _currentPartial = "正在识别…"; _asrProcessStart = System.currentTimeMillis() }
        override fun onResult(text: String) { if (_asrProcessStart > 0) _asrLatencyMs = System.currentTimeMillis() - _asrProcessStart; onAsrResult(text) }
        override fun onError(msg: String) { log("本地Whisper: $msg") }
        override fun onDownloadProgress(file: String, percent: Int) { _downloadProgress = "$file: $percent%" }
        override fun onModelReady() { _localWhisperReady = true }
    }

    private fun startLocalWhisper() {
        if (!_localWhisperReady) { log("Download Whisper model first"); return }
        _recording = true; log("开始录音 (本地Whisper ${selectedWhisperModel().label})")
        sherpaWhisperAsr.start(lifecycleScope, localWhisperCb)
    }

    private fun downloadWhisperModel() {
        if (_localWhisperDownloading) return; _localWhisperDownloading = true
        val m = selectedWhisperModel()
        lifecycleScope.launch {
            if (sherpaWhisperAsr.downloadModel(m, localWhisperCb)) {
                val ok = withContext(Dispatchers.IO) { sherpaWhisperAsr.initModel(m) }
                _localWhisperReady = ok; log(if (ok) "Whisper [${m.label}] 就绪" else "初始化失败")
            }
            _localWhisperDownloading = false; _downloadProgress = ""
        }
    }

    private fun switchWhisperModel(idx: Int) {
        if (_recording) { log("Stop recording first"); return }
        _whisperModelIdx = idx; saveInt("whisper_model_idx", idx)
        val m = selectedWhisperModel()
        if (sherpaWhisperAsr.isModelDownloaded(m)) {
            _localWhisperReady = false
            lifecycleScope.launch(Dispatchers.IO) {
                val ok = sherpaWhisperAsr.initModel(m)
                withContext(Dispatchers.Main) { _localWhisperReady = ok; log(if (ok) "切换 Whisper [${m.label}]" else "初始化失败") }
            }
        } else _localWhisperReady = false
    }

    // ===================== Streaming ASR (Iter-1) =====================

    private fun selectedStreamingModel() = SherpaStreamingAsr.StreamingModel.entries[_streamingAsrModelIdx]
    private fun selectedZhTtsModel() = VitsTts.Model.entries[_zhTtsModelIdx.coerceIn(0, VitsTts.Model.entries.lastIndex)]

    private val streamingAsrCb = object : SherpaStreamingAsr.Callback {
        override fun onPartial(text: String) { _currentPartial = text }
        override fun onResult(text: String) {
            if (_asrProcessStart > 0) _asrLatencyMs = System.currentTimeMillis() - _asrProcessStart
            onAsrResult(text)
        }
        override fun onStateChanged(ready: Boolean) { _streamingAsrReady = ready }
        override fun onDownloadProgress(file: String, percent: Int) { _streamingAsrProgress = "$file: $percent%" }
        override fun onError(message: String) { log("流式ASR: $message") }
        override fun onLanguageDetected(lang: String) {
            _detectedLang = lang
            if (_langAutoMode) {
                val newTarget = if (lang == "zh") "en" else "zh"
                if (lang != _sourceLang || newTarget != _targetLang) {
                    _sourceLang = lang; _targetLang = newTarget
                    saveKey("source_lang", lang); saveKey("target_lang", newTarget)
                    updatePipelineContext()
                    translationPipeline.clearCache()
                    autoSelectTranslationEngine()
                    log("语种检测: $lang → ${lang.uppercase()}→${newTarget.uppercase()}")
                }
            }
        }
    }

    private fun startStreamingAsr() {
        if (!_streamingAsrReady) { log("Download ASR model first"); return }
        _recording = true; _asrProcessStart = System.currentTimeMillis()
        streamingAsr.denoiserEnabled = _denoiserEnabled && _denoiserReady
        streamingAsr.languageDetectionEnabled = _langAutoMode && _lidReady
        log("开始录音 (Sherpa流式 ${selectedStreamingModel().label})")
        streamingAsr.start(lifecycleScope, streamingAsrCb)
    }

    private fun downloadStreamingAsrModel() {
        if (_streamingAsrDownloading) return
        _streamingAsrDownloading = true
        val m = selectedStreamingModel()
        lifecycleScope.launch {
            if (streamingAsr.downloadModel(m, streamingAsrCb)) {
                val ok = withContext(Dispatchers.IO) { streamingAsr.initModel(m) }
                _streamingAsrReady = ok
                log(if (ok) "流式ASR [${m.label}] 就绪" else "流式ASR初始化失败")
            }
            _streamingAsrDownloading = false; _streamingAsrProgress = ""
        }
    }

    private fun switchStreamingAsrModel(idx: Int) {
        if (_recording) { log("Stop recording first"); return }
        _streamingAsrModelIdx = idx; saveInt("streaming_asr_model_idx", idx)
        val m = selectedStreamingModel()
        if (streamingAsr.isModelDownloaded(m)) {
            _streamingAsrReady = false
            lifecycleScope.launch(Dispatchers.IO) {
                val ok = streamingAsr.initModel(m)
                withContext(Dispatchers.Main) { _streamingAsrReady = ok; log(if (ok) "切换流式ASR [${m.label}]" else "初始化失败") }
            }
        } else _streamingAsrReady = false
    }

    // ===================== Kokoro TTS (Iter-2) =====================

    /** Engine 4: Sherpa offline TTS. Falls back to Edge if current model doesn't match target lang. */
    private suspend fun speakKokoro(text: String) {
        if (!_vitsTtsReady) { speakEdge(text); return }
        try {
            val queueDepth = ttsPendingCount.get()
            val speed = when {
                queueDepth >= 4 -> 1.4f
                queueDepth >= 2 -> 1.2f
                else -> 1.0f
            }
            withContext(Dispatchers.IO) { vitsTts.speak(text, sid = _sherpaTtsSid, speed = speed) }
        } catch (e: Throwable) { log("Sherpa TTS: ${e.message}"); speakEdge(text) }
    }

    private fun downloadKokoroModel() {
        if (_kokoroDownloading) return
        _kokoroDownloading = true
        lifecycleScope.launch {
            val cb = object : KokoroTts.Callback {
                override fun onDownloadProgress(file: String, percent: Int) { _kokoroProgress = "$file: $percent%" }
                override fun onError(message: String) { log("Kokoro下载: $message") }
            }
            if (kokoroTts.downloadModel(cb)) {
                val ok = withContext(Dispatchers.IO) {
                    val init = kokoroTts.initModel()
                    if (init) kokoroTts.warmUp()
                    init
                }
                _kokoroReady = ok
                log(if (ok) "Kokoro TTS 就绪" else "Kokoro 初始化失败")
            }
            _kokoroDownloading = false; _kokoroProgress = ""
        }
    }

    // ===================== Iter-3: Denoiser Download =====================

    private fun downloadDenoiserModel() {
        if (_denoiserDownloading) return; _denoiserDownloading = true
        lifecycleScope.launch {
            val cb = object : SpeechEnhancer.Callback {
                override fun onDownloadProgress(file: String, percent: Int) { _denoiserProgress = "$file: $percent%" }
                override fun onError(message: String) { log("降噪: $message") }
            }
            if (speechEnhancer.downloadModel(cb)) {
                val ok = withContext(Dispatchers.IO) { speechEnhancer.init() }
                _denoiserReady = ok
                if (ok) { streamingAsr.speechEnhancer = speechEnhancer; streamingAsr.denoiserEnabled = _denoiserEnabled }
                log(if (ok) "AI降噪就绪" else "AI降噪初始化失败")
            }
            _denoiserDownloading = false; _denoiserProgress = ""
        }
    }

    // ===================== Iter-4: Language Detector Download =====================

    private fun downloadLidModel() {
        if (_lidDownloading) return; _lidDownloading = true
        lifecycleScope.launch {
            val cb = object : LanguageDetector.Callback {
                override fun onDownloadProgress(file: String, percent: Int) { _lidProgress = "$file: $percent%" }
                override fun onError(message: String) { log("语种模型: $message") }
            }
            if (languageDetector.downloadModel(cb)) {
                val ok = withContext(Dispatchers.IO) { languageDetector.init() }
                _lidReady = ok
                if (ok) { streamingAsr.languageDetector = languageDetector; streamingAsr.languageDetectionEnabled = _langAutoMode }
                log(if (ok) "语种识别就绪" else "语种识别初始化失败")
            }
            _lidDownloading = false; _lidProgress = ""
        }
    }

    // ===================== Media Capture =====================

    private fun requestMediaCapture() {
        if (Build.VERSION.SDK_INT < 29) { log("媒体捕获需要 Android 10+"); return }
        // Stop mic ASR to avoid input conflicts
        if (_recording) { stopAllAsr(); log("已关闭麦克风，切换到媒体输入") }
        _mediaCaptureError = ""
        // Start foreground service FIRST (required for MediaProjection)
        val svcIntent = Intent(this, MediaCaptureService::class.java)
        if (Build.VERSION.SDK_INT >= 26) startForegroundService(svcIntent) else startService(svcIntent)
        val mgr = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        requestMediaProjection.launch(mgr.createScreenCaptureIntent())
    }

    private fun stopMediaCapture() {
        val svcIntent = Intent(this, MediaCaptureService::class.java).apply {
            action = MediaCaptureService.ACTION_STOP
        }
        try { startService(svcIntent) } catch (_: Throwable) {}
        try { stopService(Intent(this, MediaCaptureService::class.java)) } catch (_: Throwable) {}
        _mediaCaptureActive = false
        _mediaCaptureStatus = ""
        _mediaSourceApp = ""
        // Close any active paragraph on media capture stop
        // media capture stopped
        log("媒体捕获已停止")
    }

    // ===================== Floating Window =====================

    private fun launchFloatingWindow() {
        if (!Settings.canDrawOverlays(this)) {
            log("请授予悬浮窗权限")
            val intent = Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName"))
            startActivity(intent)
            return
        }
        val svcIntent = Intent(this, FloatingTranslateService::class.java)
        if (Build.VERSION.SDK_INT >= 26) startForegroundService(svcIntent) else startService(svcIntent)
        log("悬浮窗已启动")
    }

    // ===================== Settings =====================

    private fun loadSettings() {
        val p = getSharedPreferences("vri_settings", Context.MODE_PRIVATE)
        _openaiKey = p.getString("openai_key", "") ?: ""
        _groqKey = p.getString("groq_key", "") ?: ""
        _whisperModelIdx = p.getInt("whisper_model_idx", 0).coerceIn(0, SherpaWhisperAsr.WhisperModel.entries.size - 1)
        _asrEngine = p.getInt("asr_engine", 0)
        _ttsEngine = p.getInt("tts_engine", 0)
        _edgeVoiceIdx = p.getInt("edge_voice_idx", 0)
        _autoSpeak = p.getBoolean("auto_speak", true)
        _enableTranslation = p.getBoolean("enable_translation", true)
        _selectedInputId = p.getInt("selected_input_id", 0)
        _selectedOutputId = p.getInt("selected_output_id", 0)
        _translationEngineType = p.getInt("translation_engine", 0)
        _deeplKey = p.getString("deepl_key", "") ?: ""
        _claudeKey = p.getString("claude_key", "") ?: ""
        _claudeTransModel = p.getString("claude_trans_model", "claude-sonnet-4-20250514") ?: "claude-sonnet-4-20250514"
        _openaiVoiceIdx = p.getInt("openai_voice_idx", 4)
        _volcanoAppId = p.getString("volcano_app_id", "") ?: ""
        _volcanoToken = p.getString("volcano_access_token", "") ?: ""
        _volcanoCluster = p.getString("volcano_cluster", com.example.myapplication1.tts.VolcanoTts.DEFAULT_CLUSTER)
            ?: com.example.myapplication1.tts.VolcanoTts.DEFAULT_CLUSTER
        _volcanoVoiceIdx = p.getInt("volcano_voice_idx", 0)
        _localServerUrl = p.getString("local_server_url", "http://192.168.1.100:11434/v1") ?: "http://192.168.1.100:11434/v1"
        _localServerModel = p.getString("local_server_model", "qwen2.5:7b") ?: "qwen2.5:7b"
        _smartFilterEnabled = p.getBoolean("smart_filter_enabled", true)
        _filterFillers = p.getBoolean("filter_fillers", true)
        _filterEcho = p.getBoolean("filter_echo", true)
        _filterNoise = p.getBoolean("filter_noise", true)
        _filterMusic = p.getBoolean("filter_music", true)
        _denoiserEnabled = p.getBoolean("denoiser_enabled", false)
        _uiLang = p.getString("ui_lang", "zh") ?: "zh"
        UiStrings.load(this, _uiLang)
        _sourceLang = p.getString("source_lang", "en") ?: "en"
        _targetLang = p.getString("target_lang", "zh") ?: "zh"
        _langAutoMode = p.getBoolean("lang_auto_mode", true)
        _refineProvider = p.getInt("refine_provider", 0)
        _refineModel = p.getString("refine_model", "llama-3.3-70b-versatile") ?: "llama-3.3-70b-versatile"
        _refineServerUrl = p.getString("refine_server_url", "http://192.168.1.100:11434/v1") ?: "http://192.168.1.100:11434/v1"
        _latencyMode = p.getInt("latency_mode", 0)
        _backgroundText = p.getString("background_text", "") ?: ""
        _domainHint = p.getString("domain_hint", "auto") ?: "auto"
        _paragraphAutoGroup = p.getBoolean("paragraph_auto_group", true)
        _paragraphMaxSegments = p.getInt("paragraph_max_segments", 8).coerceIn(1, 20)
        _paragraphSilenceMs = p.getInt("paragraph_silence_ms", 4000).coerceIn(1000, 10000)
        _streamingAsrModelIdx = p.getInt("streaming_asr_model_idx", 0).coerceIn(0, SherpaStreamingAsr.StreamingModel.entries.size - 1)
        _kokoroVoiceSid = p.getInt("kokoro_voice_sid", 0).coerceIn(0, KokoroTts.VOICES.last().first)
        _zhTtsModelIdx = p.getInt("zh_tts_model_idx", 0).coerceIn(0, VitsTts.Model.entries.lastIndex)
        _sherpaTtsSid = p.getInt("sherpa_tts_sid", 0)
        _ortProvider = AccelerationConfig.provider(this)
    }

    private fun saveKey(k: String, v: String) { getSharedPreferences("vri_settings", Context.MODE_PRIVATE).edit().putString(k, v).apply() }
    private fun saveInt(k: String, v: Int) { getSharedPreferences("vri_settings", Context.MODE_PRIVATE).edit().putInt(k, v).apply() }
    private fun saveBool(k: String, v: Boolean) { getSharedPreferences("vri_settings", Context.MODE_PRIVATE).edit().putBoolean(k, v).apply() }
    private fun saveFloat(k: String, v: Float) { getSharedPreferences("vri_settings", Context.MODE_PRIVATE).edit().putFloat(k, v).apply() }

    /**
     * Switch the ONNX Runtime execution provider (CPU/NNAPI/XNNPACK) and rebuild
     * every loaded Sherpa engine so the change takes effect.  Refuses while recording.
     * All release() calls happen on IO to avoid blocking the UI.
     */
    private fun applyOrtProvider(newProvider: String) {
        if (_recording) { log("请先停止录音再切换加速模式"); return }
        if (newProvider == _ortProvider) return
        saveKey(AccelerationConfig.KEY, newProvider)
        // Read back the *effective* provider: if user picked NNAPI on API<27,
        // AccelerationConfig.provider() returns "cpu", keeping UI honest.
        val effective = AccelerationConfig.provider(this)
        _ortProvider = effective
        if (effective != newProvider) {
            log("系统不支持 $newProvider，已自动降级为 $effective")
        } else {
            log("加速模式已切换至 $effective，正在重载模型…")
        }
        lifecycleScope.launch(Dispatchers.IO) {
            try { if (::streamingAsr.isInitialized) streamingAsr.release() } catch (_: Throwable) {}
            try { sherpaWhisperAsr.release() } catch (_: Throwable) {}
            try { if (::speechEnhancer.isInitialized) speechEnhancer.release() } catch (_: Throwable) {}
            try { if (::languageDetector.isInitialized) languageDetector.release() } catch (_: Throwable) {}
            try { if (::vitsTts.isInitialized) vitsTts.release() } catch (_: Throwable) {}
            try { if (::kokoroTts.isInitialized) kokoroTts.release() } catch (_: Throwable) {}
            try { cachedTransEngine?.close() } catch (_: Throwable) {}
            cachedTransEngine = null
            cachedTransEngineType = -1
            withContext(Dispatchers.Main) {
                _streamingAsrReady = false
                _localWhisperReady = false
                _vitsTtsReady = false
                _kokoroReady = false
                _denoiserReady = false
                _lidReady = false
                log("加速模式切换完成，下次使用将按新模式初始化")
            }
        }
    }

    private fun selectedWhisperModel() = SherpaWhisperAsr.WhisperModel.entries[_whisperModelIdx]

    // ===================== Sentence Buffer =====================

    /**
     * Entry point for all ASR results.  Applies echo suppression and smart filter,
     * then adds the utterance to the current paragraph and submits for translation.
     *
     * Design: trust ASR boundaries — each onAsrResult call is one complete utterance.
     * No re-segmentation.  Paragraph breaks are detected by silence gaps.
     */
    private fun onAsrResult(text: String) {
        log("ASR: $text")  // ← 关键日志：ASR结果到达

        // Echo suppression
        if (_recording && !_isTtsSpeaking) {
            val msSinceTts = System.currentTimeMillis() - _ttsSpeakEndTime
            val isLikelyEcho = msSinceTts < 400 && text.length < 6
            if (isLikelyEcho) {
                log("ASR回声抑制: ${text.take(30)}")
                return
            }
        }

        // Smart filter
        if (_smartFilterEnabled) {
            val config = AsrTextFilter.FilterConfig(_filterFillers, _filterEcho, _filterNoise, _filterMusic)
            val filtered = AsrTextFilter.filter(text, config)
            if (filtered == null) {
                log("ASR过滤: ${text.take(30)}")
                return
            }
            AsrTextFilter.recordText(filtered)
            addSegmentToParagraph(filtered)
        } else {
            addSegmentToParagraph(text)
        }
    }

    /**
     * Add an ASR utterance as a new segment in the current paragraph.
     * Creates a new paragraph if none exists.  Resets the silence-based paragraph break timer.
     *
     * Paragraph breaks are triggered by:
     * 1. Count-based: current paragraph has >= 8 segments (hard cap)
     * 2. Silence-based: PARAGRAPH_SILENCE_MS elapses with no new ASR result
     */
    private fun addSegmentToParagraph(text: String) {
        _currentPartial = ""
        paragraphSilenceJob?.cancel()

        // 决定是否创建新段落：关闭聚合时每句独立；启用聚合时按 max segments 切分
        // 从历史加载的段落永远只读，新句子必须新开段落
        val lastIsHistory = _paragraphs.lastOrNull()?.fromHistory == true
        val shouldCreateNewParagraph = if (!_paragraphAutoGroup || lastIsHistory) {
            true  // 关闭聚合 或 上一段来自历史 → 新起一段
        } else {
            _paragraphs.isEmpty() ||
                (_paragraphs.last().allDone && _paragraphs.last().segments.size >= _paragraphMaxSegments)
        }
        if (shouldCreateNewParagraph) {
            paragraphsAdd(Paragraph(id = _nextParagraphId++))
        }

        val paraIdx = _paragraphs.lastIndex
        val para = _paragraphs[paraIdx]
        val seqId = translationPipeline.allocateSeqId()
        val newSeg = Segment(seqId = seqId, en = text, translating = true)
        val existingZh = para.segments.filter { it.zh.isNotBlank() }.size
        paragraphsSet(paraIdx, para.copy(segments = para.segments + newSeg))

        log("提交翻译 seq=$seqId pi=$paraIdx existZh=$existingZh totalSegs=${para.segments.size + 1}: ${text.take(40)}")
        translationHistory.appendPending(seqId, text)
        translationPipeline.submitSentence(seqId, para.id, text)

        val currentParaId = para.id
        if (_paragraphAutoGroup) {
            // 启用聚合：静音阈值到后收尾段落并创建新段落
            paragraphSilenceJob = lifecycleScope.launch {
                delay(_paragraphSilenceMs.toLong())
                if (_paragraphs.isNotEmpty()) {
                    val lastPara = _paragraphs.last()
                    if (lastPara.id == currentParaId && lastPara.segments.isNotEmpty()) {
                        if (lastPara.allDone) {
                            translationPipeline.closeParagraph(currentParaId)
                        }
                        paragraphsAdd(Paragraph(id = _nextParagraphId++))
                    }
                }
            }
        } else {
            // 关闭聚合：每句立即 closeParagraph（润色按句触发）
            paragraphSilenceJob = lifecycleScope.launch {
                // 等待翻译完成再 close；最多等 10s 兜底
                val deadline = System.currentTimeMillis() + 10_000L
                while (System.currentTimeMillis() < deadline) {
                    val p = _paragraphs.find { it.id == currentParaId }
                    if (p == null || p.allDone) break
                    delay(150)
                }
                translationPipeline.closeParagraph(currentParaId)
            }
        }
    }

    // ===================== Translation Engine =====================

    private fun getTranslationEngine(): TranslationEngine {
        if (cachedTransEngineType == _translationEngineType && cachedTransEngine != null) return cachedTransEngine!!
        cachedTransEngine?.close()
        cachedTransEngine = when (_translationEngineType) {
            1 -> LLMTranslation(_openaiKey)
            2 -> LLMTranslation(_groqKey, "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile")
            3 -> DeepLTranslation(_deeplKey)
            4 -> LocalServerTranslation(_localServerUrl, _localServerModel)
            5 -> {
                val engine = OnDeviceTranslation(this, OnDeviceTranslationModel.OPUS_MT_EN_ZH)
                if (OnDeviceTranslationModel.OPUS_MT_EN_ZH.isDownloaded(this)) engine.init()
                engine
            }
            6 -> {
                val engine = OnDeviceTranslation(this, OnDeviceTranslationModel.NLLB_600M_INT8)
                if (OnDeviceTranslationModel.NLLB_600M_INT8.isDownloaded(this)) engine.init()
                engine
            }
            7 -> ClaudeTranslation(_claudeKey, _claudeTransModel)
            else -> MlKitTranslation(translator)
        }
        cachedTransEngineType = _translationEngineType
        return cachedTransEngine!!
    }

    private fun invalidateTranslationCache() {
        cachedTransEngineType = -1; cachedTransEngine?.close(); cachedTransEngine = null
        if (::translationPipeline.isInitialized) {
            translationPipeline.setEngine(getTranslationEngine())
            translationPipeline.setRefiner(buildRefiner())
            translationPipeline.clearCache()
            updatePipelineContext()
        }
    }

    /** Sync latency mode, background, domain hint, and quality engine to the pipeline. */
    private fun updatePipelineContext() {
        if (!::translationPipeline.isInitialized) return
        translationPipeline.translationContext = TranslationContext(
            background = _backgroundText,
            domainHint = _domainHint,
            latencyMode = when (_latencyMode) {
                1 -> LatencyMode.BALANCED
                2 -> LatencyMode.QUALITY
                else -> LatencyMode.REALTIME
            },
            sourceLang = _sourceLang,
            targetLang = _targetLang,
        )
        translationPipeline.setQualityEngine(buildQualityEngine())
        // Sync ASR language hint for Whisper API engines
        whisperAsr?.language = _sourceLang
    }

    /**
     * Build a quality engine for SWR dual-channel translation.
     * Picks the best available API engine that differs from the primary engine.
     * Returns null if no suitable quality engine is available or mode is REALTIME.
     */
    private fun buildQualityEngine(): TranslationEngine? {
        if (_latencyMode == 0) return null  // REALTIME — no quality path needed
        val primary = _translationEngineType
        // Prefer Claude (best), then OpenAI (high quality), then DeepL, then Groq (fast), then local
        if (primary != 7 && _claudeKey.isNotBlank())
            return ClaudeTranslation(_claudeKey, _claudeTransModel)
        if (primary != 1 && _openaiKey.isNotBlank())
            return LLMTranslation(_openaiKey)
        if (primary != 3 && _deeplKey.isNotBlank())
            return DeepLTranslation(_deeplKey)
        if (primary != 2 && _groqKey.isNotBlank())
            return LLMTranslation(_groqKey, "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile")
        if (primary != 4 && _localServerUrl.isNotBlank())
            return LocalServerTranslation(_localServerUrl, _localServerModel)
        return null
    }

    private fun buildRefiner(): TranslationRefiner? {
        return when (_refineProvider) {
            TranslationRefiner.PROVIDER_GROQ -> {
                if (_groqKey.isNotBlank()) TranslationRefiner(_groqKey, TranslationRefiner.providerBaseUrl(TranslationRefiner.PROVIDER_GROQ), _refineModel) else null
            }
            TranslationRefiner.PROVIDER_OPENAI -> {
                if (_openaiKey.isNotBlank()) TranslationRefiner(_openaiKey, TranslationRefiner.providerBaseUrl(TranslationRefiner.PROVIDER_OPENAI), _refineModel) else null
            }
            TranslationRefiner.PROVIDER_ON_DEVICE -> {
                TranslationRefiner("", TranslationRefiner.providerBaseUrl(TranslationRefiner.PROVIDER_ON_DEVICE), _refineModel)
            }
            TranslationRefiner.PROVIDER_LOCAL -> {
                if (_refineServerUrl.isNotBlank()) TranslationRefiner("", _refineServerUrl, _refineModel) else null
            }
            TranslationRefiner.PROVIDER_CLAUDE -> {
                if (_claudeKey.isNotBlank()) TranslationRefiner(_claudeKey, TranslationRefiner.providerBaseUrl(TranslationRefiner.PROVIDER_CLAUDE), _refineModel) else null
            }
            else -> null
        }
    }

    private fun invalidateOpenAiTts() { openAiTts?.close(); openAiTts = null }

    private fun translationReady() = when (_translationEngineType) {
        1 -> _openaiKey.isNotBlank()
        2 -> _groqKey.isNotBlank()
        3 -> _deeplKey.isNotBlank()
        4 -> _localServerUrl.isNotBlank()
        5 -> OnDeviceTranslationModel.OPUS_MT_EN_ZH.isDownloaded(this)
        6 -> OnDeviceTranslationModel.NLLB_600M_INT8.isDownloaded(this)
        7 -> _claudeKey.isNotBlank()
        else -> _translatorReady
    }

    // ===================== Pipeline Callback =====================

    private val pipelineCallback = object : TranslationPipeline.Callback {
        override fun onTranslationStarted(seqId: Int, en: String) {}

        override fun onTranslationResult(seqId: Int, en: String, zh: String) {
            // Already on Main thread (pipeline uses withContext(Dispatchers.Main))
            for (pi in _paragraphs.indices) {
                val para = _paragraphs[pi]
                if (para.fromHistory) continue  // history segments have stale seqIds — skip
                val si = para.segments.indexOfFirst { it.seqId == seqId }
                if (si >= 0) {
                    val newSegs = para.segments.toMutableList()
                    newSegs[si] = newSegs[si].copy(zh = zh, translating = false)
                    paragraphsSet(pi, para.copy(segments = newSegs))
                    log("翻译完成 seq=$seqId pi=$pi: $en → $zh")
                    // Update history on IO to avoid blocking Main thread / Compose recomposition
                    lifecycleScope.launch(Dispatchers.IO) {
                        translationHistory.updateZhBySeqId(seqId, zh)
                    }
                    return
                }
            }
            Log.w("VRI", "翻译结果 seq=$seqId 未匹配段落 (paragraphs=${_paragraphs.size})")
        }

        override fun onTranslationError(seqId: Int, en: String, error: String) {
            for (pi in _paragraphs.indices) {
                val para = _paragraphs[pi]
                if (para.fromHistory) continue
                val si = para.segments.indexOfFirst { it.seqId == seqId }
                if (si >= 0) {
                    val newSegs = para.segments.toMutableList()
                    newSegs[si] = newSegs[si].copy(zh = "[翻译失败]", translating = false)
                    paragraphsSet(pi, para.copy(segments = newSegs))
                    log("翻译失败: $error")
                    return
                }
            }
        }

        override fun onTtsReady(zh: String) {
            ttsPendingCount.incrementAndGet()
            _ttsQueueSize = ttsPendingCount.get()
            ttsQueue.trySend(zh)
        }

        override fun onLatencyMeasured(translationMs: Long) {
            _transLatencyMs = translationMs
        }

        override fun onCacheHit(seqId: Int) {
            _transCacheHits = translationPipeline.cacheHitCount
        }

        override fun onParagraphRefined(paragraphId: Int, refinedZh: String) {
            if (refinedZh.isNotBlank()) {
                Log.d("VRI", "Paragraph $paragraphId refined: ${refinedZh.take(80)}")
            }
        }

        override fun onTranslationUpgraded(seqId: Int, en: String, zh: String, meta: TranslationMeta) {
            // SWR quality upgrade — update segment display without triggering TTS
            for (pi in _paragraphs.indices) {
                val para = _paragraphs[pi]
                if (para.fromHistory) continue
                val si = para.segments.indexOfFirst { it.seqId == seqId }
                if (si >= 0) {
                    val seg = para.segments[si]
                    val newSegs = para.segments.toMutableList()
                    newSegs[si] = seg.copy(qualityZh = zh, route = meta.route)
                    paragraphsSet(pi, para.copy(segments = newSegs))
                    lifecycleScope.launch(Dispatchers.IO) {
                        translationHistory.upsertZhBySeqId(seqId, zh)
                    }
                    log("质量升级: ${en.take(20)} → ${zh.take(20)}")
                    return
                }
            }
        }
    }

    // ===================== Utility =====================

    private fun log(msg: String) { Log.d("VRI", msg); runOnUiThread { _logs.add(0, msg); if (_logs.size > 100) _logs.removeLastOrNull() } }
    private fun copyAssetDir(a: String, d: File) {
        if (!d.exists()) d.mkdirs()
        for (n in assets.list(a) ?: emptyArray()) { val p = "$a/$n"; val c = assets.list(p); if (c.isNullOrEmpty()) assets.open(p).use { i -> FileOutputStream(File(d, n)).use { o -> i.copyTo(o) } } else copyAssetDir(p, File(d, n)) }
    }
    private fun clearAll() {
        paragraphSilenceJob?.cancel(); paragraphSilenceJob = null
        paragraphsClear(); _nextParagraphId = 0; _currentPartial = ""
        if (::translationPipeline.isInitialized) translationPipeline.reset()
        log(S("clear_all"))
    }

    /** 加载指定会话到主界面继续对话。 */
    private fun loadSessionIntoMain(sessionId: String) {
        val session = translationHistory.allSessions().find { it.id == sessionId } ?: return
        // 如在录音则先停止
        if (_recording) {
            stopAllAsr()
            log("已停止录音（切换会话）")
        }
        if (_mediaCaptureActive) {
            stopMediaCapture()
        }
        // 切换当前会话
        translationHistory.switchToSession(sessionId)
        // 清空主界面并重建段落
        paragraphSilenceJob?.cancel(); paragraphSilenceJob = null
        paragraphsClear(); _nextParagraphId = 0; _currentPartial = ""
        if (::translationPipeline.isInitialized) translationPipeline.reset()
        // 加载 entries：每条独立成一个段落卡片（保证按句显示，避免拼成长串）
        // 新录入的句子仍按用户聚合设置处理
        session.entries.forEach { e ->
            val seg = Segment(seqId = e.seqId, en = e.en, zh = e.zh, translating = false)
            paragraphsAdd(Paragraph(
                id = _nextParagraphId++, segments = listOf(seg), fromHistory = true
            ))
        }
        // 导航回主界面
        _showHistory = false
        log("Session: ：${session.displayTitle}")
    }

    /** 新建一个空会话：清空主界面，但不删除历史。 */
    private fun onNewConversation() {
        if (_recording) {
            log("请先停止录音再新建对话")
            return
        }
        if (_mediaCaptureActive) {
            log("请先停止媒体捕获再新建对话")
            return
        }
        // 刷新历史持久化
        translationHistory.flush()
        // 清空当前 UI 状态
        paragraphSilenceJob?.cancel(); paragraphSilenceJob = null
        paragraphsClear(); _nextParagraphId = 0; _currentPartial = ""
        if (::translationPipeline.isInitialized) translationPipeline.reset()
        // 创建新会话并刷新列表
        translationHistory.newSession()
        _historySessions.clear()
        _historySessions.addAll(translationHistory.allSessions())
        log(S("new_conversation"))
    }

    /** 删除单个会话（带正确的状态同步）。 */
    private fun handleDeleteSession(sessionId: String) {
        val isCurrent = sessionId == translationHistory.currentSession()?.id
        // 如删的是当前会话且正在录音，先停止录音
        if (isCurrent && _recording) {
            stopAllAsr()
            log("已停止录音（因删除当前会话）")
        }
        // 如删的是当前会话，清空 UI 与翻译流水线状态
        if (isCurrent) {
            paragraphSilenceJob?.cancel(); paragraphSilenceJob = null
            paragraphsClear(); _nextParagraphId = 0; _currentPartial = ""
            if (::translationPipeline.isInitialized) translationPipeline.reset()
        }
        // 执行删除
        translationHistory.deleteSession(sessionId)
        // 如删的是当前会话，切换到最新会话或创建新会话
        if (isCurrent) {
            if (translationHistory.allSessions().isNotEmpty()) {
                translationHistory.continueOrNew()
            } else {
                translationHistory.newSession()
            }
        }
        _historySessions.clear()
        _historySessions.addAll(translationHistory.allSessions())
        log(S("delete"))
    }

    /** 清空所有历史（带正确的状态同步）。 */
    private fun handleClearAllSessions() {
        if (_recording) { stopAllAsr(); log("已停止录音（因清空所有历史）") }
        paragraphSilenceJob?.cancel(); paragraphSilenceJob = null
        paragraphsClear(); _nextParagraphId = 0; _currentPartial = ""
        if (::translationPipeline.isInitialized) translationPipeline.reset()
        translationHistory.clearAll()
        translationHistory.newSession()
        _historySessions.clear()
        _historySessions.addAll(translationHistory.allSessions())
        log(S("clear_all"))
    }

    private fun asrReady() = when (_asrEngine) { 0 -> _systemAsrAvailable; 1 -> _voskReady; 2, 5 -> _openaiKey.isNotBlank(); 3 -> _groqKey.isNotBlank(); 4 -> _localWhisperReady; 6 -> _streamingAsrReady; else -> false }
    private fun asrLabel(): String {
        val asr = when (_asrEngine) { 0 -> "系统识别"; 1 -> "Vosk"; 2 -> "OpenAI"; 3 -> "Groq"; 4 -> "Whisper ${selectedWhisperModel().label}"; 5 -> "GPT-4o"; 6 -> "流式 ${selectedStreamingModel().label}"; else -> "" }
        if (!_enableTranslation) return "$asr → 仅识别"
        val trans = when (_translationEngineType) { 0 -> "MLKit"; 1 -> "GPT"; 2 -> "Groq"; 3 -> "DeepL"; 4 -> "本地LLM"; 5 -> "Opus-MT"; 6 -> "NLLB"; else -> "" }
        val refine = if (_refineProvider > 0) " +润色" else ""
        return "$asr → $trans$refine"
    }

    // ===================== Compose UI =====================

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    private fun AppUI() {
        BackHandler(enabled = _showHistory) { _showHistory = false }
        if (_showHistory) {
            HistoryScreen()
            return
        }
        val drawerState = rememberDrawerState(DrawerValue.Closed)
        val scope = rememberCoroutineScope()

        ModalNavigationDrawer(
            drawerState = drawerState,
            drawerContent = { ModalDrawerSheet(Modifier.width(310.dp)) { DrawerContent(drawerState) } }
        ) {
            Scaffold(
                topBar = {
                    TopAppBar(
                        navigationIcon = { IconButton({ scope.launch { drawerState.open() } }) { Icon(Icons.Default.Menu, "菜单") } },
                        title = {
                            val curId = translationHistory.currentSession()?.id
                            val currentSessionNum = if (curId == null) 0
                            else translationHistory.allSessions().sortedBy { it.startTime }
                                .indexOfFirst { it.id == curId }
                                .let { if (it >= 0) it + 1 else 0 }
                            Column {
                                Text(S("app_title"), fontWeight = FontWeight.Bold, fontSize = 18.sp)
                                Text(
                                    if (currentSessionNum > 0) "#$currentSessionNum · ${asrLabel()}" else asrLabel(),
                                    fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                                )
                            }
                        },
                        actions = {
                            IconButton({ onNewConversation() }) {
                                Icon(Icons.Default.AddCircleOutline, S("new_conversation"))
                            }
                            IconButton({ _showHistory = !_showHistory }) {
                                Icon(Icons.Default.History, "历史",
                                    tint = if (_showHistory) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                            }
                            IconButton({ clearAll() }) { Icon(Icons.Default.ClearAll, "清空当前") }
                        },
                        colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface)
                    )
                },
                floatingActionButton = { RecordFAB(_recording) { toggleRecording() } },
                floatingActionButtonPosition = FabPosition.Center
            ) { pad ->
                // Session dialog
                if (_showSessionDialog) {
                    AlertDialog(
                        onDismissRequest = { _showSessionDialog = false; translationHistory.newSession() },
                        title = { Text(S("session_select")) },
                        text = { Text(S("session_select_msg")) },
                        confirmButton = {
                            TextButton({
                                _showSessionDialog = false
                                translationHistory.newSession()
                                _historySessions.clear()
                                _historySessions.addAll(translationHistory.allSessions())
                            }) { Text(S("new_conversation")) }
                        },
                        dismissButton = {
                            TextButton({
                                _showSessionDialog = false
                                translationHistory.continueOrNew()
                                // Load last session's entries into a paragraph
                                val lastSession = translationHistory.currentSession()
                                if (lastSession != null && lastSession.entries.isNotEmpty()) {
                                    paragraphsClear()
                                    val segs = lastSession.entries.map { e -> Segment(en = e.en, zh = e.zh, translating = false) }
                                    paragraphsAdd(Paragraph(id = _nextParagraphId++, segments = segs))
                                }
                            }) { Text(S("continue_last")) }
                        }
                    )
                }

                Column(Modifier.fillMaxSize().padding(pad)) {
                    // 状态条：ASR/翻译/TTS 三个状态点（从 TopAppBar 移出以避免拥挤）
                    Surface(
                        color = MaterialTheme.colorScheme.surface,
                        tonalElevation = 0.dp,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(
                            Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.End
                        ) {
                            CompactStatusDot(asrReady(), "ASR")
                            Spacer(Modifier.width(10.dp))
                            CompactStatusDot(translationReady() && _enableTranslation, "翻译")
                            Spacer(Modifier.width(10.dp))
                            CompactStatusDot(_ttsReady && _ttsLangOk, "TTS")
                        }
                    }
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.2f))

                    // ---- Iter-4: 语言方向栏 (始终可见) ----
                    LanguageBar()
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.2f))

                    // 延迟指标栏
                    if (_recording || _ttsQueueSize > 0) {
                        val dimColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                        Surface(color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f), modifier = Modifier.fillMaxWidth()) {
                            Column(Modifier.padding(horizontal = 16.dp, vertical = 4.dp)) {
                                // 流水线延迟
                                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                    if (_asrLatencyMs > 0) Text("ASR: ${_asrLatencyMs}ms", fontSize = 10.sp, color = dimColor)
                                    if (_transLatencyMs > 0) Text("Trans: ${_transLatencyMs}ms", fontSize = 10.sp, color = dimColor)
                                    if (_refineLatencyMs > 0) Text("Refine: ${_refineLatencyMs}ms", fontSize = 10.sp, color = dimColor)
                                    if (_ttsLatencyMs > 0) Text("TTS: ${_ttsLatencyMs}ms", fontSize = 10.sp, color = dimColor)
                                    if (_ttsQueueSize > 0) Text("Queue: $_ttsQueueSize", fontSize = 10.sp, color = Color(0xFFEF5350))
                                    if (_ttsSpeedPct > 0) Text("Speed: +${_ttsSpeedPct}%", fontSize = 10.sp, color = MaterialTheme.colorScheme.primary)
                                    if (_transCacheHits > 0) Text("Cache: $_transCacheHits", fontSize = 10.sp, color = Color(0xFF4CAF50))
                                }
                                // 设备指标
                                if (_cpuUsagePct > 0f || _memoryUsageMB > 0) {
                                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                        Text("CPU: ${"%.1f".format(_cpuUsagePct)}%", fontSize = 10.sp, color = dimColor)
                                        Text("Mem: ${_memoryUsageMB}MB", fontSize = 10.sp, color = dimColor)
                                        if (_batteryPct > 0) Text("Bat: ${_batteryPct}%", fontSize = 10.sp, color = if (_batteryPct <= 15) Color(0xFFEF5350) else dimColor)
                                        if (_batteryTempC > 0f) Text("Temp: ${"%.1f".format(_batteryTempC)}°C", fontSize = 10.sp, color = if (_batteryTempC >= 40f) Color(0xFFEF5350) else dimColor)
                                    }
                                }
                            }
                        }
                    }
                    // 媒体捕获状态栏
                    if (_mediaCaptureActive || _mediaCaptureError.isNotBlank()) {
                        val isError = _mediaCaptureError.isNotBlank() && !_mediaCaptureActive
                        val bgColor = if (isError) Color(0xFFEF5350).copy(alpha = 0.12f) else Color(0xFF4CAF50).copy(alpha = 0.12f)
                        val iconColor = if (isError) Color(0xFFEF5350) else Color(0xFF4CAF50)
                        Surface(color = bgColor, modifier = Modifier.fillMaxWidth()) {
                            Row(Modifier.padding(horizontal = 16.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                                Icon(if (isError) Icons.Default.ErrorOutline else Icons.Default.Audiotrack, null, Modifier.size(16.dp), tint = iconColor)
                                Spacer(Modifier.width(8.dp))
                                Column(Modifier.weight(1f)) {
                                    if (isError) {
                                        Text(S("capture_failed"), fontSize = 13.sp, color = Color(0xFFD32F2F))
                                        Text(_mediaCaptureError, fontSize = 11.sp, color = Color(0xFFD32F2F).copy(alpha = 0.7f))
                                    } else {
                                        Text(
                                            if (_mediaCaptureStatus.isNotBlank()) _mediaCaptureStatus else S("capture_active"),
                                            fontSize = 13.sp, color = Color(0xFF388E3C)
                                        )
                                        Row {
                                            Text("Input: Media", fontSize = 11.sp, color = Color(0xFF388E3C).copy(alpha = 0.7f))
                                            if (_mediaSourceApp.isNotBlank()) {
                                                Text(" · $_mediaSourceApp", fontSize = 11.sp, color = Color(0xFF388E3C).copy(alpha = 0.7f))
                                            }
                                        }
                                    }
                                }
                                if (_mediaCaptureActive) {
                                    TextButton({ stopMediaCapture() }) { Text(S("stop_capture"), fontSize = 12.sp) }
                                } else if (isError) {
                                    TextButton({ _mediaCaptureError = "" }) { Text(S("close"), fontSize = 12.sp) }
                                }
                            }
                        }
                    }
                    ConversationArea(Modifier.weight(1f))
                }
            }
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    private fun HistoryScreen() {
        Scaffold(
            topBar = {
                TopAppBar(
                    navigationIcon = {
                        IconButton({ _showHistory = false }) {
                            Icon(Icons.Default.ArrowBack, "返回")
                        }
                    },
                    title = {
                        Text(S("history"), fontWeight = FontWeight.Bold, fontSize = 18.sp)
                    },
                    actions = {
                        IconButton({
                            onNewConversation()
                            _showHistory = false
                        }) {
                            Icon(Icons.Default.AddCircleOutline, S("new_conversation"))
                        }
                        if (_historySessions.isNotEmpty()) {
                            IconButton({ _showClearAllDialog = true }) {
                                Icon(Icons.Default.Delete, "清空全部",
                                     tint = Color(0xFFEF5350).copy(alpha = 0.8f))
                            }
                        }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = MaterialTheme.colorScheme.surface)
                )
            }
        ) { pad ->
            Box(Modifier.fillMaxSize().padding(pad)) {
                HistoryArea(Modifier.fillMaxSize())
            }
        }
    }

    @Composable
    private fun StatusDot(ok: Boolean, label: String) {
        val c by animateColorAsState(if (ok) Color(0xFF4CAF50) else Color(0xFFBDBDBD))
        Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.padding(horizontal = 4.dp)) {
            Box(Modifier.size(8.dp).clip(CircleShape).background(c))
            Text(label, fontSize = 9.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
        }
    }

    /** 横向紧凑状态点：圆点 + 标签左右排列，占用宽度小。 */
    @Composable
    private fun CompactStatusDot(ok: Boolean, label: String) {
        val c by animateColorAsState(if (ok) Color(0xFF4CAF50) else Color(0xFFBDBDBD))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(7.dp).clip(CircleShape).background(c))
            Spacer(Modifier.width(4.dp))
            Text(label, fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.55f))
        }
    }

    // ===================== Drawer =====================

    @Composable
    private fun DrawerContent(drawerState: DrawerState) {
        // 非主页时拦截返回键返回主页
        BackHandler(enabled = drawerState.isOpen && _settingsScreen != SettingsScreen.Main) {
            _settingsScreen = SettingsScreen.Main
        }

        Column(Modifier.fillMaxHeight()) {
            // 顶部 header：主页显示标题，子页显示返回按钮
            if (_settingsScreen == SettingsScreen.Main) {
                Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Text(S("settings"), fontWeight = FontWeight.Bold, fontSize = 22.sp)
                }
            } else {
                Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    IconButton({ _settingsScreen = SettingsScreen.Main }) {
                        Icon(Icons.Default.ArrowBack, "返回")
                    }
                    Text(
                        when (_settingsScreen) {
                            SettingsScreen.General -> S("general")
                            SettingsScreen.Voice -> S("voice")
                            SettingsScreen.Translation -> S("translation")
                            SettingsScreen.Advanced -> S("advanced")
                            else -> ""
                        },
                        fontWeight = FontWeight.Bold, fontSize = 18.sp
                    )
                }
                HorizontalDivider()
            }

            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp)) {
                Spacer(Modifier.height(8.dp))
                when (_settingsScreen) {
                    SettingsScreen.Main -> SettingsMainPage()
                    SettingsScreen.General -> GeneralSettingsPage()
                    SettingsScreen.Voice -> VoiceSettingsPage()
                    SettingsScreen.Translation -> TranslationSettingsPage()
                    SettingsScreen.Advanced -> AdvancedSettingsPage()
                }
                Spacer(Modifier.height(40.dp))
            }
        }
    }

    @Composable
    private fun SettingsMainPage() {
        CategoryEntry(S("general"), S("general_desc"), Icons.Default.Tune, SettingsScreen.General)
        CategoryEntry(S("voice"), S("voice_desc"), Icons.Default.Mic, SettingsScreen.Voice)
        CategoryEntry(S("translation"), S("translation_desc"), Icons.Default.Language, SettingsScreen.Translation)
        CategoryEntry(S("advanced"), S("advanced_desc"), Icons.Default.Settings, SettingsScreen.Advanced)
        Spacer(Modifier.height(20.dp))
        HorizontalDivider()
        Spacer(Modifier.height(12.dp))
        Text("About", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
        Text("RealTimeTranslateTTS", fontSize = 11.sp,
             color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f))
    }

    @Composable
    private fun CategoryEntry(
        title: String,
        subtitle: String,
        icon: ImageVector,
        screen: SettingsScreen
    ) {
        Surface(
            onClick = { _settingsScreen = screen },
            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
            shape = RoundedCornerShape(12.dp),
            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
        ) {
            Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(icon, null, Modifier.size(24.dp), tint = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(title, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
                    Text(subtitle, fontSize = 11.sp,
                         color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
                }
                Icon(Icons.Default.KeyboardArrowRight, null,
                     Modifier.size(18.dp),
                     tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f))
            }
        }
    }

    // ==================== 分类页面 ====================

    @Composable
    private fun GeneralSettingsPage() {
        // 0) 界面语言
        CollapsibleSection(S("ui_language"), Icons.Default.Language, defaultExpanded = true) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf("zh" to "中文", "en" to "English").forEach { (code, label) ->
                    Surface(
                        onClick = {
                            _uiLang = code; UiStrings.load(this@MainActivity, code); saveKey("ui_lang", code)
                        },
                        shape = RoundedCornerShape(8.dp),
                        color = if (_uiLang == code) MaterialTheme.colorScheme.primaryContainer
                               else MaterialTheme.colorScheme.surfaceVariant,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text(label, fontSize = 14.sp, fontWeight = if (_uiLang == code) FontWeight.Bold else FontWeight.Normal,
                            modifier = Modifier.padding(vertical = 10.dp),
                            textAlign = androidx.compose.ui.text.style.TextAlign.Center)
                    }
                }
            }
        }
        // 1) 基本设置
        CollapsibleSection(S("basic_settings"), Icons.Default.Settings, defaultExpanded = true) {
            SettingSwitch(S("enable_translation"), _enableTranslation) {
                _enableTranslation = it
                saveBool("enable_translation", it)
                translationPipeline.enableTranslation = it
            }
            SettingSwitch(S("auto_speak"), _autoSpeak) { _autoSpeak = it; saveBool("auto_speak", it) }
        }
        // 2) 段落聚合
        @Suppress("DEPRECATION")
        CollapsibleSection(S("paragraph_grouping"), Icons.Default.List, defaultExpanded = true) {
            Text(S("group_off_hint"), fontSize = 10.sp,
                 color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.45f),
                 modifier = Modifier.padding(bottom = 6.dp))

            SettingSwitch(S("auto_group"), _paragraphAutoGroup) {
                _paragraphAutoGroup = it
                saveBool("paragraph_auto_group", it)
            }

            AnimatedVisibility(_paragraphAutoGroup) {
                Column(Modifier.padding(start = 8.dp, top = 6.dp)) {
                    Text("${S("max_segments_per_para")}: $_paragraphMaxSegments", fontSize = 12.sp,
                         color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f))
                    Slider(
                        value = _paragraphMaxSegments.toFloat(),
                        onValueChange = { _paragraphMaxSegments = it.toInt().coerceIn(1, 20) },
                        onValueChangeFinished = { saveInt("paragraph_max_segments", _paragraphMaxSegments) },
                        valueRange = 1f..20f, steps = 18
                    )
                    val secs = _paragraphSilenceMs / 1000f
                    Text("${S("silence_break")}: ${"%.1f".format(secs)}s", fontSize = 12.sp,
                         color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f))
                    Slider(
                        value = _paragraphSilenceMs.toFloat(),
                        onValueChange = { _paragraphSilenceMs = it.toInt().coerceIn(1000, 10000) },
                        onValueChangeFinished = { saveInt("paragraph_silence_ms", _paragraphSilenceMs) },
                        valueRange = 1000f..10000f, steps = 17
                    )
                }
            }

            AnimatedVisibility(!_paragraphAutoGroup) {
                Surface(color = Color(0xFFFFF3CD), shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth().padding(top = 6.dp)) {
                    Text(S("group_warn"),
                         fontSize = 10.sp, color = Color(0xFF856404),
                         modifier = Modifier.padding(8.dp))
                }
            }
        }
        // 3) 语种检测 & 翻译方向
        LanguageDetectionSection()
    }

    @Composable
    private fun LanguageDetectionSection() {
        CollapsibleSection(S("lang_detection"), Icons.Default.Translate) {
            SettingSwitch(S("auto_detect_lang"), _langAutoMode) {
                _langAutoMode = it; saveBool("lang_auto_mode", it)
                streamingAsr.languageDetectionEnabled = it && _lidReady
            }
            Text(S("auto_detect_hint"), fontSize = 10.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f))

            AnimatedVisibility(!_langAutoMode) {
                Column(Modifier.padding(start = 8.dp, top = 6.dp)) {
                    Text(S("manual_lang_pair"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("${S("source_label")}: ", fontSize = 12.sp)
                        listOf("en", "zh", "ja", "ko", "fr", "de").forEach { lang ->
                            TextButton(onClick = { _sourceLang = lang; saveKey("source_lang", lang); updatePipelineContext(); autoSelectTranslationEngine() },
                                contentPadding = PaddingValues(horizontal = 6.dp, vertical = 2.dp),
                                colors = if (_sourceLang == lang) ButtonDefaults.textButtonColors(
                                    containerColor = MaterialTheme.colorScheme.primaryContainer) else ButtonDefaults.textButtonColors()
                            ) { Text(lang.uppercase(), fontSize = 11.sp) }
                        }
                    }
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("${S("target_label")}: ", fontSize = 12.sp)
                        listOf("zh", "en", "ja", "ko", "fr", "de").forEach { lang ->
                            TextButton(onClick = { _targetLang = lang; saveKey("target_lang", lang); updatePipelineContext(); autoSelectTranslationEngine() },
                                contentPadding = PaddingValues(horizontal = 6.dp, vertical = 2.dp),
                                colors = if (_targetLang == lang) ButtonDefaults.textButtonColors(
                                    containerColor = MaterialTheme.colorScheme.primaryContainer) else ButtonDefaults.textButtonColors()
                            ) { Text(lang.uppercase(), fontSize = 11.sp) }
                        }
                    }
                }
            }

            Spacer(Modifier.height(6.dp))
            when {
                _lidReady -> Text(S("lid_ready"), fontSize = 12.sp, color = Color(0xFF4CAF50))
                _lidDownloading -> {
                    Text(_lidProgress.ifBlank { "下载中…" }, fontSize = 12.sp, color = MaterialTheme.colorScheme.primary)
                    LinearProgressIndicator(Modifier.fillMaxWidth())
                }
                else -> Button(onClick = { downloadLidModel() },
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp)) {
                    Icon(Icons.Default.Download, null, Modifier.size(14.dp)); Spacer(Modifier.width(4.dp))
                    Text("${S("download_lid")} (~103MB)", fontSize = 12.sp)
                }
            }
            if (_detectedLang.isNotBlank()) {
                Spacer(Modifier.height(4.dp))
                Text("${S("current_detection")}: ${_detectedLang.uppercase()} → ${_sourceLang.uppercase()}→${_targetLang.uppercase()}", fontSize = 11.sp,
                    color = MaterialTheme.colorScheme.primary)
            }
        }
    }

    @Composable
    private fun VoiceSettingsPage() {
        LegacySectionsForCategory(SettingsScreen.Voice)
    }

    @Composable
    private fun TranslationSettingsPage() {
        LegacySectionsForCategory(SettingsScreen.Translation)
    }

    @Composable
    private fun AdvancedSettingsPage() {
        LegacySectionsForCategory(SettingsScreen.Advanced)
    }

    // ==================== 原有配置区（按分类过滤） ====================

    @Composable
    private fun LegacySectionsForCategory(category: SettingsScreen) {
        // 2) 语音识别
        if (category == SettingsScreen.Voice) CollapsibleSection(S("asr_engine"), Icons.Default.Mic) {
                AsrOption(0, S("android_system"), if (_systemAsrAvailable) "${S("need_network")} · ${S("multilingual")}" else S("unavailable"))
                AsrOption(1, S("vosk_offline"), "${S("offline")} · EN")
                AsrOption(2, "OpenAI Whisper", "API · ${S("multi_auto_detect")}")
                AsrOption(3, "Groq Whisper", "API · ${S("multilingual")} · ${S("free_quota")}")
                AsrOption(4, S("local_whisper"), "${S("offline")} · ${S("multilingual")} · ${S("high_accuracy")}")
                AsrOption(5, S("gpt4o_voice"), "API · ${S("multilingual")} · ${S("high_accuracy")}")
                AsrOption(6, S("sherpa_streaming"), "Zipformer · ${S("by_model_lang")} · ${S("recommended")}")

                AnimatedVisibility(_asrEngine == 1) { VoskModelPanel() }
                AnimatedVisibility(_asrEngine == 2 || _asrEngine == 5) { ApiKeyInput("OpenAI API Key", _openaiKey) { _openaiKey = it; saveKey("openai_key", it); invalidateTranslationCache() } }
                AnimatedVisibility(_asrEngine == 3) { ApiKeyInput("Groq API Key", _groqKey) { _groqKey = it; saveKey("groq_key", it); invalidateTranslationCache() } }
                AnimatedVisibility(_asrEngine == 4) { WhisperModelSelector() }
                AnimatedVisibility(_asrEngine == 6) { StreamingAsrModelPanel() }
            }

            // 3) 翻译引擎
            if (category == SettingsScreen.Translation) CollapsibleSection(S("trans_engine"), Icons.Default.Language) {
                TransOption(0, S("mlkit_offline"), S("mlkit_desc"))
                TransOption(1, S("openai_gpt"), S("high_quality"))
                TransOption(2, S("groq_llm"), S("groq_desc"))
                TransOption(3, S("deepl"), S("deepl_desc"))
                TransOption(7, S("claude_api"), S("claude_desc"))
                TransOption(4, S("local_server"), S("local_server_desc"))
                TransOption(5, S("opus_mt"), S("opus_mt_desc"))
                TransOption(6, S("nllb"), S("nllb_desc"))

                AnimatedVisibility(_translationEngineType == 4) {
                    Column(Modifier.padding(start = 4.dp, top = 6.dp)) {
                        OutlinedTextField(_localServerUrl,
                            { _localServerUrl = it; saveKey("local_server_url", it); invalidateTranslationCache() },
                            label = { Text(S("server_url"), fontSize = 12.sp) }, singleLine = true,
                            placeholder = { Text("http://192.168.1.100:11434/v1") },
                            modifier = Modifier.fillMaxWidth(), textStyle = LocalTextStyle.current.copy(fontSize = 13.sp))
                        Spacer(Modifier.height(4.dp))
                        OutlinedTextField(_localServerModel,
                            { _localServerModel = it; saveKey("local_server_model", it); invalidateTranslationCache() },
                            label = { Text(S("model_name"), fontSize = 12.sp) }, singleLine = true,
                            placeholder = { Text("qwen2.5:7b") },
                            modifier = Modifier.fillMaxWidth(), textStyle = LocalTextStyle.current.copy(fontSize = 13.sp))
                        Text("Ollama, LM Studio, OpenAI-compatible API", fontSize = 10.sp,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
                            modifier = Modifier.padding(top = 4.dp))
                    }
                }

                // Offline model download UI
                AnimatedVisibility(_translationEngineType == 5 || _translationEngineType == 6) {
                    OfflineTransModelPanel()
                }

                AnimatedVisibility(_translationEngineType == 3) {
                    ApiKeyInput("DeepL API Key", _deeplKey) { _deeplKey = it; saveKey("deepl_key", it); invalidateTranslationCache() }
                }
                AnimatedVisibility(_translationEngineType == 7) {
                    Column(Modifier.padding(start = 4.dp, top = 6.dp)) {
                        ApiKeyInput("Claude API Key", _claudeKey) { _claudeKey = it; saveKey("claude_key", it); invalidateTranslationCache() }
                        Spacer(Modifier.height(6.dp))
                        Text(S("trans_model"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                        ClaudeTranslation.MODELS.forEach { preset ->
                            SmallRadio("${preset.label}  ${preset.desc}", _claudeTransModel == preset.id) {
                                _claudeTransModel = preset.id; saveKey("claude_trans_model", preset.id); invalidateTranslationCache()
                            }
                        }
                        if (_claudeKey.isBlank()) {
                            Text(S("need_claude_key_xapi"), fontSize = 11.sp, color = Color(0xFFEF5350),
                                modifier = Modifier.padding(start = 8.dp, top = 4.dp))
                        }
                    }
                }
                if (_translationEngineType == 1 && _openaiKey.isBlank()) {
                    Text(S("need_openai_key"), fontSize = 11.sp, color = Color(0xFFEF5350),
                        modifier = Modifier.padding(start = 8.dp, top = 4.dp))
                }
                if (_translationEngineType == 2 && _groqKey.isBlank()) {
                    Text(S("need_groq_key"), fontSize = 11.sp, color = Color(0xFFEF5350),
                        modifier = Modifier.padding(start = 8.dp, top = 4.dp))
                }

            }

            // 3.3) 翻译增强（延迟模式、背景信息、领域词库）
            if (category == SettingsScreen.Translation) CollapsibleSection(S("trans_enhance"), Icons.Default.Tune) {
                Text(S("trans_enhance_desc"), fontSize = 10.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.45f),
                    modifier = Modifier.padding(bottom = 6.dp))

                // Latency mode selector
                Text(S("trans_mode"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf(S("realtime") to 0, S("balanced") to 1, S("quality") to 2).forEach { (label, mode) ->
                        FilterChip(
                            selected = _latencyMode == mode,
                            onClick = { _latencyMode = mode; saveInt("latency_mode", mode); updatePipelineContext() },
                            label = { Text(label, fontSize = 12.sp) }
                        )
                    }
                }
                Text(
                    when (_latencyMode) {
                        1 -> "先出快译，后台出优译（超时回退）"
                        2 -> "先出快译，后台出优译（不超时）"
                        else -> "仅快速翻译，最低延迟"
                    },
                    fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
                )

                Spacer(Modifier.height(10.dp))

                // Domain selector
                Text(S("domain_glossary"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                @OptIn(ExperimentalLayoutApi::class)
                FlowRow(Modifier.fillMaxWidth().padding(vertical = 4.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf(
                        "auto" to "自动", "general" to S("general_domain"), "meeting" to S("meeting"),
                        "medical" to S("medical"), "customer_support" to S("customer_support"), "game" to S("game")
                    ).forEach { (domain, label) ->
                        FilterChip(
                            selected = _domainHint == domain,
                            onClick = { _domainHint = domain; saveKey("domain_hint", domain); updatePipelineContext() },
                            label = { Text(label, fontSize = 12.sp) }
                        )
                    }
                }

                Spacer(Modifier.height(10.dp))

                // Background input
                Text(S("background_context"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                Text(S("background_hint"), fontSize = 10.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f))
                OutlinedTextField(
                    _backgroundText,
                    onValueChange = { if (it.length <= 300) { _backgroundText = it; saveKey("background_text", it); updatePipelineContext() } },
                    modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
                    textStyle = LocalTextStyle.current.copy(fontSize = 12.sp),
                    placeholder = { Text(S("background_placeholder"), fontSize = 12.sp) },
                    minLines = 2, maxLines = 4,
                    supportingText = { Text("${_backgroundText.length}/300", fontSize = 10.sp) }
                )
                if (_backgroundText.isNotBlank()) {
                    TextButton(onClick = { _backgroundText = ""; saveKey("background_text", ""); updatePipelineContext() },
                        modifier = Modifier.align(Alignment.End)) {
                        Text(S("clear_background"), fontSize = 12.sp)
                    }
                }
            }

            // 3.4) 术语库管理
            @Suppress("DEPRECATION")
            if (category == SettingsScreen.Translation) CollapsibleSection(S("glossary_mgmt"), Icons.Default.LibraryBooks) {
                Text(S("glossary_mgmt_desc"), fontSize = 10.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.45f),
                    modifier = Modifier.padding(bottom = 6.dp))

                // Built-in glossaries
                Text(S("builtin_glossary"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                GlossaryManager.availableDomains.forEach { (domain, label) ->
                    val count = GlossaryManager.getTerms(domain).size
                    Text("  $label ($count 条)", fontSize = 11.sp,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
                }

                Spacer(Modifier.height(10.dp))

                // Downloadable sources
                Text(S("downloadable_glossary"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                GlossaryManager.registrySources.forEach { source ->
                    val downloaded = GlossaryManager.isSourceDownloaded(source.sourceId)
                    Row(Modifier.fillMaxWidth().padding(vertical = 2.dp), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(source.name, fontSize = 12.sp)
                            Text("${source.license} · ${source.trustLevel}", fontSize = 10.sp,
                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f))
                        }
                        if (downloaded) {
                            Text(S("downloaded_label"), fontSize = 10.sp, color = Color(0xFF4CAF50))
                        } else {
                            TextButton(
                                onClick = {
                                    _glossaryDownloading = true
                                    lifecycleScope.launch {
                                        val count = GlossaryManager.downloadSource(source.sourceId)
                                        _glossaryDownloading = false
                                        _glossaryImportResult = if (count > 0) "下载成功: ${count} 条" else "下载失败"
                                    }
                                },
                                enabled = !_glossaryDownloading
                            ) { Text(if (_glossaryDownloading) S("downloading") else S("download"), fontSize = 11.sp) }
                        }
                    }
                }

                Spacer(Modifier.height(10.dp))

                // User upload
                Text(S("custom_glossary"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                Text(S("import_csv_hint"), fontSize = 10.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f))

                Row(Modifier.fillMaxWidth().padding(top = 4.dp), verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    // Domain selector for import
                    var expanded by remember { mutableStateOf(false) }
                    Box {
                        OutlinedButton(onClick = { expanded = true }) {
                            Text(GlossaryManager.getLabel(_glossaryImportDomain), fontSize = 11.sp)
                        }
                        DropdownMenu(expanded, onDismissRequest = { expanded = false }) {
                            listOf("general", "meeting", "medical", "customer_support", "game").forEach { d ->
                                DropdownMenuItem(
                                    text = { Text(GlossaryManager.getLabel(d)) },
                                    onClick = { _glossaryImportDomain = d; expanded = false }
                                )
                            }
                        }
                    }
                    Button(onClick = { glossaryFilePicker.launch("text/*") }) {
                        Text(S("select_file"), fontSize = 11.sp)
                    }
                }

                // Imported glossaries list
                val imported = GlossaryManager.importedGlossaries
                if (imported.isNotEmpty()) {
                    Spacer(Modifier.height(8.dp))
                    Text(S("imported"), fontSize = 11.sp, fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                    imported.forEach { ug ->
                        Row(Modifier.fillMaxWidth().padding(vertical = 2.dp), verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Text("${ug.fileName} (${ug.entryCount} 条)", fontSize = 11.sp)
                                Text("领域: ${GlossaryManager.getLabel(ug.domain)}", fontSize = 10.sp,
                                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f))
                            }
                            IconButton(onClick = { GlossaryManager.deleteUserGlossary(ug.id) }, Modifier.size(28.dp)) {
                                Icon(Icons.Default.Delete, S("delete"), Modifier.size(16.dp),
                                    tint = Color(0xFFEF5350))
                            }
                        }
                    }
                }

                // Import result feedback
                if (_glossaryImportResult.isNotBlank()) {
                    Spacer(Modifier.height(4.dp))
                    val isError = _glossaryImportResult.contains("失败") || _glossaryImportResult.contains("错误")
                    Text(_glossaryImportResult, fontSize = 11.sp,
                        color = if (isError) Color(0xFFEF5350) else Color(0xFF4CAF50))
                }
            }

            // 3.5) AI 润色
            if (category == SettingsScreen.Translation) CollapsibleSection(S("ai_refine"), Icons.Default.AutoAwesome) {
                Text(S("refine_desc"), fontSize = 10.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.45f),
                    modifier = Modifier.padding(bottom = 6.dp))

                RefineProviderOption(TranslationRefiner.PROVIDER_OFF, S("refine_off"), S("refine_no_refine"))
                RefineProviderOption(TranslationRefiner.PROVIDER_GROQ, "Groq API", S("refine_groq"))
                RefineProviderOption(TranslationRefiner.PROVIDER_OPENAI, "OpenAI API", S("refine_openai"))
                RefineProviderOption(TranslationRefiner.PROVIDER_CLAUDE, "Claude API", S("refine_claude"))
                RefineProviderOption(TranslationRefiner.PROVIDER_ON_DEVICE, S("refine_on_device"), S("refine_on_device_desc"))
                RefineProviderOption(TranslationRefiner.PROVIDER_LOCAL, S("refine_local"), S("refine_local_desc"))

                // Key warnings
                if (_refineProvider == TranslationRefiner.PROVIDER_GROQ && _groqKey.isBlank()) {
                    Text(S("need_groq_key"), fontSize = 11.sp, color = Color(0xFFEF5350),
                        modifier = Modifier.padding(start = 8.dp, top = 4.dp))
                }
                if (_refineProvider == TranslationRefiner.PROVIDER_OPENAI && _openaiKey.isBlank()) {
                    Text(S("need_openai_key"), fontSize = 11.sp, color = Color(0xFFEF5350),
                        modifier = Modifier.padding(start = 8.dp, top = 4.dp))
                }
                if (_refineProvider == TranslationRefiner.PROVIDER_CLAUDE && _claudeKey.isBlank()) {
                    Text(S("need_claude_key"), fontSize = 11.sp, color = Color(0xFFEF5350),
                        modifier = Modifier.padding(start = 8.dp, top = 4.dp))
                }

                // Model selection per provider
                AnimatedVisibility(_refineProvider == TranslationRefiner.PROVIDER_CLAUDE) {
                    Column(Modifier.padding(start = 8.dp, top = 6.dp)) {
                        Text(S("model_select"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                        TranslationRefiner.CLAUDE_MODELS.forEach { preset ->
                            SmallRadio("${preset.label}  ${preset.desc}", _refineModel == preset.id) {
                                _refineModel = preset.id; saveKey("refine_model", preset.id); invalidateTranslationCache()
                            }
                        }
                    }
                }
                AnimatedVisibility(_refineProvider == TranslationRefiner.PROVIDER_GROQ) {
                    Column(Modifier.padding(start = 8.dp, top = 6.dp)) {
                        Text(S("model_select"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                        TranslationRefiner.GROQ_MODELS.forEach { preset ->
                            SmallRadio("${preset.label}  ${preset.desc}", _refineModel == preset.id) {
                                _refineModel = preset.id; saveKey("refine_model", preset.id); invalidateTranslationCache()
                            }
                        }
                    }
                }
                AnimatedVisibility(_refineProvider == TranslationRefiner.PROVIDER_OPENAI) {
                    Column(Modifier.padding(start = 8.dp, top = 6.dp)) {
                        Text(S("model_select"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                        TranslationRefiner.OPENAI_MODELS.forEach { preset ->
                            SmallRadio("${preset.label}  ${preset.desc}", _refineModel == preset.id) {
                                _refineModel = preset.id; saveKey("refine_model", preset.id); invalidateTranslationCache()
                            }
                        }
                    }
                }
                AnimatedVisibility(_refineProvider == TranslationRefiner.PROVIDER_ON_DEVICE) {
                    Column(Modifier.padding(start = 8.dp, top = 6.dp)) {
                        Text(S("phone_models"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                        TranslationRefiner.ON_DEVICE_MODELS.forEach { preset ->
                            SmallRadio("${preset.label}  ${preset.desc}", _refineModel == preset.id) {
                                _refineModel = preset.id; saveKey("refine_model", preset.id); invalidateTranslationCache()
                            }
                        }
                        Spacer(Modifier.height(6.dp))
                        Surface(
                            color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.3f),
                            shape = RoundedCornerShape(8.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Column(Modifier.padding(10.dp)) {
                                Text(S("deploy_steps"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                                Text("1. Install Termux (F-Droid)", fontSize = 11.sp)
                                Text("2. pkg install ollama", fontSize = 11.sp, fontStyle = FontStyle.Italic)
                                Text("3. ollama serve &", fontSize = 11.sp, fontStyle = FontStyle.Italic)
                                Text("4. ollama pull ${_refineModel}", fontSize = 11.sp, fontStyle = FontStyle.Italic)
                                Text(S("model_auto_connect"), fontSize = 10.sp,
                                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f),
                                    modifier = Modifier.padding(top = 4.dp))
                                Text(S("qwen_recommend"), fontSize = 10.sp,
                                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
                            }
                        }
                        Spacer(Modifier.height(4.dp))
                        OutlinedTextField(_refineModel,
                            { _refineModel = it; saveKey("refine_model", it); invalidateTranslationCache() },
                            label = { Text("自定义模型名称", fontSize = 12.sp) }, singleLine = true,
                            modifier = Modifier.fillMaxWidth(), textStyle = LocalTextStyle.current.copy(fontSize = 13.sp))
                    }
                }
                AnimatedVisibility(_refineProvider == TranslationRefiner.PROVIDER_LOCAL) {
                    Column(Modifier.padding(start = 8.dp, top = 6.dp)) {
                        Text(S("common_models"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                        TranslationRefiner.LOCAL_MODELS.forEach { preset ->
                            SmallRadio("${preset.label}  ${preset.desc}", _refineModel == preset.id) {
                                _refineModel = preset.id; saveKey("refine_model", preset.id); invalidateTranslationCache()
                            }
                        }
                        Spacer(Modifier.height(6.dp))
                        OutlinedTextField(_refineServerUrl,
                            { _refineServerUrl = it; saveKey("refine_server_url", it); invalidateTranslationCache() },
                            label = { Text(S("server_url"), fontSize = 12.sp) }, singleLine = true,
                            placeholder = { Text("http://192.168.1.100:11434/v1") },
                            modifier = Modifier.fillMaxWidth(), textStyle = LocalTextStyle.current.copy(fontSize = 13.sp))
                        Spacer(Modifier.height(4.dp))
                        OutlinedTextField(_refineModel,
                            { _refineModel = it; saveKey("refine_model", it); invalidateTranslationCache() },
                            label = { Text("自定义模型名称", fontSize = 12.sp) }, singleLine = true,
                            modifier = Modifier.fillMaxWidth(), textStyle = LocalTextStyle.current.copy(fontSize = 13.sp))
                        Text("Ollama, LM Studio, OpenAI-compatible API", fontSize = 10.sp,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
                            modifier = Modifier.padding(top = 4.dp))
                    }
                }
            }

            // 4) 语音合成
            if (category == SettingsScreen.Voice) CollapsibleSection(S("tts_engine"), Icons.AutoMirrored.Filled.VolumeUp) {
                TtsOption(0, S("edge_neural"), S("ms_near_human"))
                TtsOption(1, S("system_tts"), S("system_tts_desc"))
                TtsOption(2, S("google_tts"), S("basic_quality"))
                TtsOption(3, "OpenAI TTS", S("high_quality"))
                TtsOption(4, S("sherpa_offline_tts"), S("zh_offline_tts"))
                TtsOption(5, S("volcano_tts"), S("volcano_tts_desc"))
                AnimatedVisibility(_ttsEngine == 3) {
                    Column(Modifier.padding(start = 12.dp, top = 6.dp)) {
                        Text("语音角色", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                        OpenAiTts.VOICES.forEachIndexed { i, (_, l) ->
                            SmallRadio(l, _openaiVoiceIdx == i) { _openaiVoiceIdx = i; saveInt("openai_voice_idx", i); log("OpenAI语音: $l") }
                        }
                        if (_openaiKey.isBlank()) {
                            Text("需设置 OpenAI API Key", fontSize = 11.sp, color = Color(0xFFEF5350),
                                modifier = Modifier.padding(start = 8.dp, top = 4.dp))
                        }
                    }
                }
                AnimatedVisibility(_ttsEngine == 0) {
                    Column(Modifier.padding(start = 12.dp, top = 6.dp)) {
                        val voices = EdgeTts.voicesForLang(_targetLang)
                        Text("语音角色 (${_targetLang.uppercase()})", fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                        voices.forEachIndexed { i, (_, l) ->
                            SmallRadio(l, _edgeVoiceIdx == i) { _edgeVoiceIdx = i; saveInt("edge_voice_idx", i); log("语音: $l") }
                        }
                    }
                }
                AnimatedVisibility(_ttsEngine == 4) { KokoroTtsPanel() }
                AnimatedVisibility(_ttsEngine == 5) { VolcanoTtsPanel() }
            }

            // 5) 音频设备
            if (category == SettingsScreen.Voice) CollapsibleSection(S("audio_device"), Icons.Default.Headphones) {
                Text(S("input_device"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                SmallRadio(S("default_mic"), _selectedInputId == 0 && !_mediaCaptureActive) {
                    if (_mediaCaptureActive) { stopMediaCapture() }
                    _selectedInputId = 0; saveInt("selected_input_id", 0)
                }
                _inputDevices.forEach { d ->
                    SmallRadio(deviceLabel(d), _selectedInputId == d.id && !_mediaCaptureActive) {
                        if (_mediaCaptureActive) { stopMediaCapture() }
                        _selectedInputId = d.id; saveInt("selected_input_id", d.id)
                    }
                }
                // Media audio source option
                if (Build.VERSION.SDK_INT >= 29) {
                    val mediaLabel = if (_mediaCaptureActive && _mediaSourceApp.isNotBlank())
                        "媒体音频 ($_mediaSourceApp)" else "媒体音频 (系统播放)"
                    Row(
                        Modifier.fillMaxWidth().clip(RoundedCornerShape(6.dp))
                            .clickable {
                                if (_mediaCaptureActive) { stopMediaCapture() }
                                else { requestMediaCapture() }
                            }
                            .background(if (_mediaCaptureActive) MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.3f) else Color.Transparent)
                            .padding(horizontal = 8.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        RadioButton(_mediaCaptureActive, {
                            if (_mediaCaptureActive) { stopMediaCapture() }
                            else { requestMediaCapture() }
                        }, Modifier.size(18.dp))
                        Spacer(Modifier.width(6.dp))
                        Column {
                            Text(mediaLabel, fontSize = 12.sp, fontWeight = if (_mediaCaptureActive) FontWeight.SemiBold else FontWeight.Normal)
                            Text(S("media_audio_hint"), fontSize = 10.sp,
                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.45f))
                        }
                    }
                }
                Spacer(Modifier.height(10.dp))
                Text(S("output_device"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                SmallRadio(S("default_device"), _selectedOutputId == 0) { _selectedOutputId = 0; saveInt("selected_output_id", 0) }
                _outputDevices.forEach { d ->
                    SmallRadio(deviceLabel(d), _selectedOutputId == d.id) { _selectedOutputId = d.id; saveInt("selected_output_id", d.id) }
                }
                Spacer(Modifier.height(6.dp))
                TextButton(onClick = { refreshAudioDevices(); log("设备列表已刷新") }) {
                    Icon(Icons.Default.Refresh, null, Modifier.size(14.dp))
                    Spacer(Modifier.width(4.dp))
                    Text(S("refresh_devices"), fontSize = 12.sp)
                }
            }

            // Iter-3: AI 降噪
            if (category == SettingsScreen.Voice) CollapsibleSection(S("ai_denoise"), Icons.Default.GraphicEq) {
                SettingSwitch(S("enable_denoise"), _denoiserEnabled) {
                    _denoiserEnabled = it; saveBool("denoiser_enabled", it)
                    streamingAsr.denoiserEnabled = it && _denoiserReady
                }
                Text(S("denoise_hint"), fontSize = 10.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f))
                Spacer(Modifier.height(4.dp))
                when {
                    _denoiserReady -> Text(S("denoise_ready"), fontSize = 12.sp, color = Color(0xFF4CAF50))
                    _denoiserDownloading -> {
                        Text(_denoiserProgress.ifBlank { "下载中…" }, fontSize = 12.sp, color = MaterialTheme.colorScheme.primary)
                        LinearProgressIndicator(Modifier.fillMaxWidth())
                    }
                    else -> Button(onClick = { downloadDenoiserModel() },
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp)) {
                        Icon(Icons.Default.Download, null, Modifier.size(14.dp)); Spacer(Modifier.width(4.dp))
                        Text("${S("download_denoise")} (~535KB)", fontSize = 12.sp)
                    }
                }
            }

            // 6) 智能过滤
            if (category == SettingsScreen.Voice) CollapsibleSection(S("smart_filter"), Icons.Default.FilterList) {
                SettingSwitch(S("enable_filter"), _smartFilterEnabled) {
                    _smartFilterEnabled = it; saveBool("smart_filter_enabled", it)
                    if (!it) AsrTextFilter.reset()
                }
                AnimatedVisibility(_smartFilterEnabled) {
                    Column(Modifier.padding(start = 8.dp)) {
                        SettingSwitch(S("filter_fillers"), _filterFillers) { _filterFillers = it; saveBool("filter_fillers", it) }
                        SettingSwitch(S("filter_echo"), _filterEcho) {
                            _filterEcho = it; saveBool("filter_echo", it)
                            if (!it) AsrTextFilter.reset()
                        }
                        SettingSwitch(S("filter_noise"), _filterNoise) { _filterNoise = it; saveBool("filter_noise", it) }
                        SettingSwitch(S("filter_music"), _filterMusic) { _filterMusic = it; saveBool("filter_music", it) }
                    }
                }
                Text("智能忽略语气词、回声、噪音和音乐，提高翻译质量",
                    fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
                    modifier = Modifier.padding(top = 4.dp))
            }

            // 7) 媒体转译
            if (category == SettingsScreen.Advanced) CollapsibleSection(S("media_capture"), Icons.Default.Audiotrack) {
                Text(
                    S("capture_desc"),
                    fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                )
                Spacer(Modifier.height(8.dp))
                if (Build.VERSION.SDK_INT < 29) {
                    Text(S("sys_ver_unsupported"), fontSize = 12.sp, color = Color(0xFFEF5350))
                } else if (_mediaCaptureActive) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(S("capturing_active"), fontSize = 13.sp, color = Color(0xFF4CAF50))
                        Spacer(Modifier.weight(1f))
                        TextButton({ stopMediaCapture() }) { Text(S("stop_capture")) }
                    }
                } else {
                    Button(onClick = { requestMediaCapture() },
                        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 6.dp)) {
                        Icon(Icons.Default.PlayArrow, null, Modifier.size(16.dp))
                        Spacer(Modifier.width(6.dp))
                        Text(S("start_capture"), fontSize = 13.sp)
                    }
                }
            }

            // 7.5) GPU / NPU 加速
            if (category == SettingsScreen.Advanced) CollapsibleSection(S("gpu_accel"), Icons.Default.Memory) {
                Text(
                    S("gpu_desc"),
                    fontSize = 11.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.55f)
                )
                Spacer(Modifier.height(8.dp))
                val providers = listOf(
                    Triple(AccelerationConfig.CPU, S("gpu_cpu"), S("gpu_cpu_desc")),
                    Triple(AccelerationConfig.NNAPI, S("gpu_nnapi"), S("gpu_nnapi_desc")),
                    Triple(AccelerationConfig.XNNPACK, S("gpu_xnnpack"), S("gpu_xnnpack_desc")),
                )
                providers.forEach { (value, label, desc) ->
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth()
                            .clickable(enabled = !_recording) { applyOrtProvider(value) }
                            .padding(vertical = 4.dp)
                    ) {
                        RadioButton(
                            selected = _ortProvider == value,
                            onClick = { applyOrtProvider(value) },
                            enabled = !_recording,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(Modifier.width(8.dp))
                        Column(Modifier.weight(1f)) {
                            Text(
                                label, fontSize = 13.sp,
                                fontWeight = if (_ortProvider == value) FontWeight.SemiBold else FontWeight.Normal
                            )
                            Text(
                                desc, fontSize = 10.sp,
                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.45f)
                            )
                        }
                    }
                }
                if (_recording) {
                    Text(
                        S("recording_no_switch"),
                        fontSize = 10.sp,
                        color = Color(0xFFEF5350),
                        modifier = Modifier.padding(top = 4.dp)
                    )
                } else {
                    Text(
                        S("switch_reload_hint"),
                        fontSize = 10.sp,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
                        modifier = Modifier.padding(top = 4.dp)
                    )
                }
            }

            // 8) 悬浮窗
            if (category == SettingsScreen.Advanced) CollapsibleSection(S("floating_mode"), Icons.Default.Layers) {
                Text(
                    S("floating_desc"),
                    fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                )
                Spacer(Modifier.height(8.dp))
                if (FloatingTranslateService.isRunning) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(S("floating_running"), fontSize = 13.sp, color = Color(0xFF4CAF50))
                        Spacer(Modifier.weight(1f))
                        TextButton({ stopService(Intent(this@MainActivity, FloatingTranslateService::class.java)) }) { Text(S("stop_capture")) }
                    }
                } else {
                    Button(onClick = { launchFloatingWindow() },
                        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 6.dp)) {
                        Icon(Icons.Default.Layers, null, Modifier.size(16.dp))
                        Spacer(Modifier.width(6.dp))
                        Text(S("launch_floating"), fontSize = 13.sp)
                    }
                }
            }

            // 9) API 密钥管理
            if (category == SettingsScreen.Advanced) CollapsibleSection(S("api_key_mgmt"), Icons.Default.VpnKey) {
                ApiKeyManagerPanel()
            }

            // 10) API 连通测试
            if (category == SettingsScreen.Advanced) CollapsibleSection(S("api_test"), Icons.Default.Wifi) {
                ApiTestPanel()
            }

            // 11) 日志
            @Suppress("DEPRECATION")
            if (category == SettingsScreen.Advanced) CollapsibleSection(S("system_log"), Icons.Default.List) {
                Column(
                    Modifier.fillMaxWidth().heightIn(max = 200.dp)
                        .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(8.dp)).padding(8.dp)
                ) {
                    if (_logs.isEmpty()) Text(S("no_log"), fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f))
                    else _logs.take(30).forEach { Text(it, fontSize = 11.sp, maxLines = 2, overflow = TextOverflow.Ellipsis, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)) }
                }
            }

    }

    // ---- Collapsible Section ----

    @Composable
    private fun CollapsibleSection(
        title: String,
        icon: ImageVector,
        defaultExpanded: Boolean = false,
        content: @Composable ColumnScope.() -> Unit
    ) {
        var expanded by remember { mutableStateOf(defaultExpanded) }
        val rotation by animateFloatAsState(if (expanded) 180f else 0f)

        Column(Modifier.padding(vertical = 4.dp)) {
            Surface(
                onClick = { expanded = !expanded },
                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                shape = RoundedCornerShape(10.dp)
            ) {
                Row(
                    Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(icon, null, Modifier.size(20.dp), tint = MaterialTheme.colorScheme.primary)
                    Spacer(Modifier.width(10.dp))
                    Text(title, fontSize = 15.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                    Icon(Icons.Default.KeyboardArrowDown, null, Modifier.size(20.dp).rotate(rotation),
                        tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f))
                }
            }
            AnimatedVisibility(visible = expanded) {
                Column(Modifier.padding(start = 8.dp, top = 6.dp, end = 4.dp, bottom = 4.dp)) { content() }
            }
        }
    }

    // ---- Composable helpers ----

    @Composable
    private fun SettingSwitch(title: String, checked: Boolean, on: (Boolean) -> Unit) {
        Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(title, fontSize = 14.sp, modifier = Modifier.weight(1f))
            Switch(checked, on)
        }
    }

    @Composable
    private fun AsrOption(id: Int, title: String, desc: String) {
        Row(
            Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp))
                .clickable { if (_recording) log("Stop recording first") else { _asrEngine = id; saveInt("asr_engine", id) } }
                .background(if (_asrEngine == id) MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.35f) else Color.Transparent)
                .padding(horizontal = 8.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            RadioButton(_asrEngine == id, { if (_recording) log("Stop recording first") else { _asrEngine = id; saveInt("asr_engine", id) } }, Modifier.size(20.dp))
            Spacer(Modifier.width(8.dp))
            Column {
                Text(title, fontSize = 14.sp, fontWeight = if (_asrEngine == id) FontWeight.SemiBold else FontWeight.Normal)
                Text(desc, fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
            }
        }
    }

    @Composable
    private fun TtsOption(id: Int, title: String, desc: String) {
        Row(
            Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp))
                .clickable { _ttsEngine = id; saveInt("tts_engine", id) }
                .background(if (_ttsEngine == id) MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.35f) else Color.Transparent)
                .padding(horizontal = 8.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            RadioButton(_ttsEngine == id, { _ttsEngine = id; saveInt("tts_engine", id) }, Modifier.size(20.dp))
            Spacer(Modifier.width(8.dp))
            Column {
                Text(title, fontSize = 14.sp, fontWeight = if (_ttsEngine == id) FontWeight.SemiBold else FontWeight.Normal)
                Text(desc, fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
            }
        }
    }

    /**
     * Language pairs each translation engine supports.
     * null = supports all languages (LLM-based engines).
     */
    private fun engineSupportedPairs(engineId: Int): Set<String>? = when (engineId) {
        0 -> setOf("en-zh")              // MLKit: EN→ZH only (hardcoded translator)
        5 -> setOf("en-zh")              // Opus-MT: EN→ZH only
        6 -> setOf("en-zh")              // NLLB: hardcoded eng_Latn→zho_Hans
        1, 2, 4 -> null                  // LLM engines: any language (dynamic prompt)
        3 -> null                         // DeepL: most languages (dynamic API params)
        else -> null
    }

    /** Check if an engine supports the current language pair. */
    private fun engineSupportsCurrentPair(engineId: Int): Boolean {
        val pairs = engineSupportedPairs(engineId) ?: return true
        return "${_sourceLang}-${_targetLang}" in pairs
    }

    /** Auto-select best engine for current language pair if current engine is incompatible. */
    private fun autoSelectTranslationEngine() {
        if (engineSupportsCurrentPair(_translationEngineType)) return
        // Find first compatible engine: prefer LLM (2=Groq, 1=OpenAI), then DeepL, then NLLB
        val preferred = listOf(2, 1, 3, 6, 4, 0, 5)
        val best = preferred.firstOrNull { engineSupportsCurrentPair(it) } ?: 2
        _translationEngineType = best
        saveInt("translation_engine", best)
        invalidateTranslationCache()
        log("翻译引擎自动切换为 $best (当前语言对不支持)")
    }

    @Composable
    private fun TransOption(id: Int, title: String, desc: String) {
        val supported = engineSupportsCurrentPair(id)
        val alpha = if (supported) 1f else 0.35f
        Row(
            Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp))
                .clickable(enabled = supported) {
                    _translationEngineType = id; saveInt("translation_engine", id); invalidateTranslationCache()
                }
                .background(if (_translationEngineType == id) MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.35f) else Color.Transparent)
                .padding(horizontal = 8.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            RadioButton(_translationEngineType == id, onClick = {
                if (supported) { _translationEngineType = id; saveInt("translation_engine", id); invalidateTranslationCache() }
            }, Modifier.size(20.dp), enabled = supported)
            Spacer(Modifier.width(8.dp))
            Column {
                Text(title, fontSize = 14.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = alpha),
                    fontWeight = if (_translationEngineType == id) FontWeight.SemiBold else FontWeight.Normal)
                Text(
                    desc + if (!supported) " · 不支持 ${_sourceLang.uppercase()}→${_targetLang.uppercase()}" else "",
                    fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f * alpha)
                )
            }
        }
    }

    @Composable
    private fun RefineProviderOption(id: Int, title: String, desc: String) {
        val selected = _refineProvider == id
        Row(
            Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp))
                .clickable {
                    _refineProvider = id; saveInt("refine_provider", id)
                    // Set default model for provider
                    if (id != TranslationRefiner.PROVIDER_OFF) {
                        val defaultModel = TranslationRefiner.defaultModel(id)
                        _refineModel = defaultModel; saveKey("refine_model", defaultModel)
                    }
                    invalidateTranslationCache()
                }
                .background(if (selected) MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.35f) else Color.Transparent)
                .padding(horizontal = 8.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            RadioButton(selected, {
                _refineProvider = id; saveInt("refine_provider", id)
                if (id != TranslationRefiner.PROVIDER_OFF) {
                    val defaultModel = TranslationRefiner.defaultModel(id)
                    _refineModel = defaultModel; saveKey("refine_model", defaultModel)
                }
                invalidateTranslationCache()
            }, Modifier.size(20.dp))
            Spacer(Modifier.width(8.dp))
            Column {
                Text(title, fontSize = 14.sp, fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal)
                Text(desc, fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
            }
        }
    }

    @Composable
    private fun SmallRadio(label: String, selected: Boolean, onClick: () -> Unit) {
        Row(
            Modifier.fillMaxWidth().clip(RoundedCornerShape(6.dp)).clickable(onClick = onClick)
                .background(if (selected) MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.3f) else Color.Transparent)
                .padding(horizontal = 8.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            RadioButton(selected, onClick, Modifier.size(18.dp))
            Spacer(Modifier.width(6.dp))
            Text(label, fontSize = 12.sp)
        }
    }

    @Composable
    private fun ApiKeyInput(label: String, value: String, on: (String) -> Unit) {
        Column(Modifier.padding(start = 4.dp, top = 6.dp)) {
            OutlinedTextField(value, on, label = { Text(label, fontSize = 12.sp) },
                visualTransformation = PasswordVisualTransformation(), singleLine = true,
                modifier = Modifier.fillMaxWidth(), textStyle = LocalTextStyle.current.copy(fontSize = 13.sp))
            Text(if (value.isBlank()) S("enter_api_key") else S("already_set"), fontSize = 11.sp,
                color = if (value.isBlank()) Color(0xFFEF5350) else Color(0xFF4CAF50),
                modifier = Modifier.padding(start = 4.dp, top = 2.dp))
        }
    }

    // ===================== API Key Manager =====================

    @Composable
    private fun ApiKeyManagerPanel() {
        Column {
            Text(S("configured_keys"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
            Spacer(Modifier.height(8.dp))

            // Show existing keys
            data class KeyEntry(val name: String, val prefKey: String, val value: String, val onSet: (String) -> Unit)
            val keys = listOf(
                KeyEntry("OpenAI", "openai_key", _openaiKey) { _openaiKey = it; saveKey("openai_key", it); invalidateTranslationCache() },
                KeyEntry("Groq", "groq_key", _groqKey) { _groqKey = it; saveKey("groq_key", it); invalidateTranslationCache() },
                KeyEntry("DeepL", "deepl_key", _deeplKey) { _deeplKey = it; saveKey("deepl_key", it); invalidateTranslationCache() },
                KeyEntry("Claude", "claude_key", _claudeKey) { _claudeKey = it; saveKey("claude_key", it); invalidateTranslationCache() },
                KeyEntry("火山引擎 App ID", "volcano_app_id", _volcanoAppId) { _volcanoAppId = it; saveKey("volcano_app_id", it) },
                KeyEntry("火山引擎 Token", "volcano_access_token", _volcanoToken) { _volcanoToken = it; saveKey("volcano_access_token", it) },
            )

            keys.forEach { entry ->
                Surface(
                    color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp)
                ) {
                    Row(
                        Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(entry.name, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                            if (entry.value.isNotBlank()) {
                                val masked = entry.value.take(4) + "****" + entry.value.takeLast(4)
                                Text(masked, fontSize = 11.sp, color = Color(0xFF4CAF50))
                            } else {
                                Text(S("not_set"), fontSize = 11.sp, color = Color(0xFFEF5350))
                            }
                        }
                        if (entry.value.isNotBlank()) {
                            TextButton(onClick = {
                                entry.onSet("")
                                log("已删除 ${entry.name} API Key")
                            }, contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp)) {
                                Text(S("delete"), fontSize = 12.sp, color = Color(0xFFEF5350))
                            }
                        }
                    }
                }
            }

            // Server configs
            Spacer(Modifier.height(10.dp))
            Text(S("server_config"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
            Spacer(Modifier.height(4.dp))
            Surface(
                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp)
            ) {
                Column(Modifier.padding(10.dp)) {
                    Text(S("local_trans_server"), fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                    OutlinedTextField(
                        _localServerUrl,
                        { _localServerUrl = it; saveKey("local_server_url", it); invalidateTranslationCache() },
                        label = { Text("URL", fontSize = 11.sp) }, singleLine = true,
                        placeholder = { Text("http://192.168.1.100:11434/v1") },
                        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                        textStyle = androidx.compose.ui.text.TextStyle(fontSize = 12.sp)
                    )
                    OutlinedTextField(
                        _localServerModel,
                        { _localServerModel = it; saveKey("local_server_model", it); invalidateTranslationCache() },
                        label = { Text(S("model_label"), fontSize = 11.sp) }, singleLine = true,
                        placeholder = { Text("qwen2.5:7b") },
                        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                        textStyle = androidx.compose.ui.text.TextStyle(fontSize = 12.sp)
                    )
                }
            }
            Surface(
                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp)
            ) {
                Column(Modifier.padding(10.dp)) {
                    Text(S("refine_server_label"), fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                    OutlinedTextField(
                        _refineServerUrl,
                        { _refineServerUrl = it; saveKey("refine_server_url", it); invalidateTranslationCache() },
                        label = { Text("URL", fontSize = 11.sp) }, singleLine = true,
                        placeholder = { Text("http://192.168.1.100:11434/v1") },
                        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                        textStyle = androidx.compose.ui.text.TextStyle(fontSize = 12.sp)
                    )
                    OutlinedTextField(
                        _refineModel,
                        { _refineModel = it; saveKey("refine_model", it); invalidateTranslationCache() },
                        label = { Text(S("model_label"), fontSize = 11.sp) }, singleLine = true,
                        placeholder = { Text("llama-3.3-70b-versatile") },
                        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                        textStyle = androidx.compose.ui.text.TextStyle(fontSize = 12.sp)
                    )
                }
            }
            // Volcano cluster
            Surface(
                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp)
            ) {
                Column(Modifier.padding(10.dp)) {
                    Text(S("volcano_cluster_label"), fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                    OutlinedTextField(
                        _volcanoCluster,
                        { _volcanoCluster = it.ifBlank { com.example.myapplication1.tts.VolcanoTts.DEFAULT_CLUSTER }; saveKey("volcano_cluster", _volcanoCluster) },
                        label = { Text("Cluster", fontSize = 11.sp) }, singleLine = true,
                        placeholder = { Text("volcano_tts") },
                        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                        textStyle = androidx.compose.ui.text.TextStyle(fontSize = 12.sp)
                    )
                }
            }

            // Add new key section
            Spacer(Modifier.height(12.dp))
            Text(S("add_modify_key"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
            Spacer(Modifier.height(4.dp))

            var selectedKeyType by remember { mutableStateOf(0) }
            val keyTypes = listOf("OpenAI", "Groq", "DeepL", "Claude", "火山 App ID", "火山 Token")
            @OptIn(ExperimentalLayoutApi::class)
            FlowRow(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                keyTypes.forEachIndexed { i, label ->
                    FilterChip(
                        selected = selectedKeyType == i,
                        onClick = { selectedKeyType = i },
                        label = { Text(label, fontSize = 10.sp) },
                    )
                }
            }
            Spacer(Modifier.height(4.dp))
            OutlinedTextField(
                _newKeyValue, { _newKeyValue = it },
                label = { Text("${keyTypes[selectedKeyType]} Key", fontSize = 12.sp) },
                visualTransformation = PasswordVisualTransformation(),
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
                textStyle = LocalTextStyle.current.copy(fontSize = 13.sp)
            )
            Spacer(Modifier.height(6.dp))
            Button(
                onClick = {
                    if (_newKeyValue.isNotBlank()) {
                        when (selectedKeyType) {
                            0 -> { _openaiKey = _newKeyValue; saveKey("openai_key", _newKeyValue) }
                            1 -> { _groqKey = _newKeyValue; saveKey("groq_key", _newKeyValue) }
                            2 -> { _deeplKey = _newKeyValue; saveKey("deepl_key", _newKeyValue) }
                            3 -> { _claudeKey = _newKeyValue; saveKey("claude_key", _newKeyValue) }
                            4 -> { _volcanoAppId = _newKeyValue; saveKey("volcano_app_id", _newKeyValue) }
                            5 -> { _volcanoToken = _newKeyValue; saveKey("volcano_access_token", _newKeyValue) }
                        }
                        invalidateTranslationCache()
                        log("已保存 ${keyTypes[selectedKeyType]}")
                        _newKeyValue = ""
                    }
                },
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 6.dp),
                enabled = _newKeyValue.isNotBlank()
            ) {
                Icon(Icons.Default.Done, null, Modifier.size(16.dp))
                Spacer(Modifier.width(6.dp))
                Text(S("save"), fontSize = 13.sp)
            }
        }
    }

    // ===================== API Test Panel =====================

    @Composable
    private fun ApiTestPanel() {
        Column {
            Text(S("api_test_desc"),
                fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.45f))
            Spacer(Modifier.height(8.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                Button(
                    onClick = { runAllApiTests() },
                    enabled = !_apiTestRunning,
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 6.dp)
                ) {
                    if (_apiTestRunning) {
                        CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp, color = MaterialTheme.colorScheme.onPrimary)
                    } else {
                        Icon(Icons.Default.PlayArrow, null, Modifier.size(16.dp))
                    }
                    Spacer(Modifier.width(6.dp))
                    Text(if (_apiTestRunning) S("testing_progress") else S("test_all"), fontSize = 13.sp)
                }
                if (_apiTestProgress.isNotBlank()) {
                    Spacer(Modifier.width(10.dp))
                    Text(_apiTestProgress, fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
                }
            }

            // Individual test buttons
            Spacer(Modifier.height(8.dp))
            Text(S("individual_test"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
            Spacer(Modifier.height(4.dp))

            @Composable
            fun SingleTestButton(label: String, available: Boolean, onClick: () -> Unit) {
                Row(
                    Modifier.fillMaxWidth().padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    TextButton(
                        onClick = onClick,
                        enabled = !_apiTestRunning && available,
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 2.dp)
                    ) {
                        Icon(Icons.Default.PlayCircle, null, Modifier.size(14.dp))
                        Spacer(Modifier.width(4.dp))
                        Text(label, fontSize = 12.sp)
                    }
                    if (!available) {
                        Text(S("not_configured"), fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.35f))
                    }
                }
            }

            SingleTestButton(S("test_openai"), _openaiKey.isNotBlank()) {
                runSingleApiTest("openai")
            }
            SingleTestButton(S("test_groq"), _groqKey.isNotBlank()) {
                runSingleApiTest("groq")
            }
            SingleTestButton(S("test_deepl"), _deeplKey.isNotBlank()) {
                runSingleApiTest("deepl")
            }
            SingleTestButton(S("test_claude"), _claudeKey.isNotBlank()) {
                runSingleApiTest("claude")
            }
            SingleTestButton("Edge TTS", true) {
                runSingleApiTest("edge")
            }
            SingleTestButton(S("test_google_tts"), true) {
                runSingleApiTest("google_tts")
            }
            SingleTestButton(S("test_volcano"), _volcanoAppId.isNotBlank() && _volcanoToken.isNotBlank()) {
                runSingleApiTest("volcano")
            }
            SingleTestButton(S("test_local"), _localServerUrl.isNotBlank()) {
                runSingleApiTest("local")
            }

            // Results
            if (_apiTestResults.isNotEmpty()) {
                Spacer(Modifier.height(12.dp))
                Text(S("test_results"), fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.height(6.dp))

                _apiTestResults.forEach { result ->
                    ApiTestResultCard(result)
                    Spacer(Modifier.height(6.dp))
                }
            }
        }
    }

    @Composable
    private fun ApiTestResultCard(result: ApiTestManager.ApiTestResult) {
        val bgColor = if (result.overallSuccess)
            Color(0xFF4CAF50).copy(alpha = 0.08f)
        else
            Color(0xFFEF5350).copy(alpha = 0.08f)
        val borderColor = if (result.overallSuccess) Color(0xFF4CAF50) else Color(0xFFEF5350)

        Surface(
            color = bgColor,
            shape = RoundedCornerShape(8.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(Modifier.padding(10.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        if (result.overallSuccess) Icons.Default.CheckCircle else Icons.Default.Error,
                        null, Modifier.size(18.dp),
                        tint = borderColor
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        result.apiName, fontSize = 14.sp, fontWeight = FontWeight.SemiBold,
                        color = borderColor
                    )
                    Spacer(Modifier.weight(1f))
                    Text(
                        if (result.overallSuccess) "通过" else "失败",
                        fontSize = 12.sp, fontWeight = FontWeight.Bold,
                        color = borderColor
                    )
                }
                Spacer(Modifier.height(6.dp))

                result.steps.forEach { step ->
                    Row(
                        Modifier.fillMaxWidth().padding(vertical = 1.dp),
                        verticalAlignment = Alignment.Top
                    ) {
                        val stepIcon = if (step.success) "✓" else "✗"
                        val stepColor = if (step.success) Color(0xFF4CAF50) else Color(0xFFEF5350)
                        Text(stepIcon, fontSize = 12.sp, color = stepColor, fontWeight = FontWeight.Bold,
                            modifier = Modifier.width(16.dp))
                        Text("${step.step}: ", fontSize = 11.sp, fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f))
                        Column(Modifier.weight(1f)) {
                            Text(step.detail, fontSize = 11.sp,
                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                        }
                        if (step.durationMs > 0) {
                            Text("${step.durationMs}ms", fontSize = 10.sp,
                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.35f))
                        }
                    }
                }

                // Show blocked step info
                val failed = result.failedStep
                if (failed != null) {
                    Spacer(Modifier.height(4.dp))
                    Surface(
                        color = Color(0xFFEF5350).copy(alpha = 0.1f),
                        shape = RoundedCornerShape(6.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(Modifier.padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Warning, null, Modifier.size(14.dp), tint = Color(0xFFD32F2F))
                            Spacer(Modifier.width(6.dp))
                            Text("${S("blocked_at")}: ${failed.step} — ${failed.detail}",
                                fontSize = 11.sp, color = Color(0xFFD32F2F))
                        }
                    }
                }
            }
        }
    }

    private fun runAllApiTests() {
        if (_apiTestRunning) return
        _apiTestRunning = true
        _apiTestResults.clear()
        _apiTestProgress = "准备中..."

        lifecycleScope.launch {
            val tests = mutableListOf<suspend () -> ApiTestManager.ApiTestResult>()

            // Always test free services
            tests.add { apiTestManager.testEdgeTts() }
            tests.add { apiTestManager.testGoogleTts() }

            // Test configured APIs
            if (_openaiKey.isNotBlank()) {
                tests.add { apiTestManager.testOpenAiTranslation(_openaiKey) }
                tests.add { apiTestManager.testOpenAiWhisper(_openaiKey) }
                tests.add { apiTestManager.testOpenAiTts(_openaiKey) }
            }
            if (_groqKey.isNotBlank()) {
                tests.add { apiTestManager.testGroqTranslation(_groqKey) }
                tests.add { apiTestManager.testGroqWhisper(_groqKey) }
            }
            if (_deeplKey.isNotBlank()) {
                tests.add { apiTestManager.testDeepL(_deeplKey) }
            }
            if (_claudeKey.isNotBlank()) {
                tests.add { apiTestManager.testClaudeTranslation(_claudeKey) }
            }
            if (_volcanoAppId.isNotBlank() && _volcanoToken.isNotBlank()) {
                tests.add { apiTestManager.testVolcanoTts(_volcanoAppId, _volcanoToken, _volcanoCluster) }
            }
            if (_localServerUrl.isNotBlank()) {
                tests.add { apiTestManager.testLocalServer(_localServerUrl, _localServerModel) }
            }

            val semaphore = Semaphore(3)
            val completed = java.util.concurrent.atomic.AtomicInteger(0)
            val total = tests.size

            val deferreds = tests.map { test ->
                async(Dispatchers.IO) {
                    semaphore.acquire()
                    try {
                        test()
                    } finally {
                        semaphore.release()
                        completed.incrementAndGet()
                    }
                }
            }

            for (deferred in deferreds) {
                try {
                    val result = deferred.await()
                    _apiTestResults.add(result)
                } catch (e: Throwable) {
                    log("API测试异常: ${e.message}")
                }
                _apiTestProgress = "${completed.get()}/$total"
            }
            _apiTestRunning = false
            _apiTestProgress = "完成 (${_apiTestResults.count { it.overallSuccess }}/${_apiTestResults.size} 通过)"
            log("API测试完成: ${_apiTestResults.count { it.overallSuccess }}/${_apiTestResults.size} 通过")
        }
    }

    private fun runSingleApiTest(type: String) {
        if (_apiTestRunning) return
        _apiTestRunning = true
        _apiTestProgress = "测试中..."

        lifecycleScope.launch {
            try {
                val results = mutableListOf<ApiTestManager.ApiTestResult>()
                when (type) {
                    "openai" -> {
                        results.add(apiTestManager.testOpenAiTranslation(_openaiKey))
                        results.add(apiTestManager.testOpenAiWhisper(_openaiKey))
                        results.add(apiTestManager.testOpenAiTts(_openaiKey))
                    }
                    "groq" -> {
                        results.add(apiTestManager.testGroqTranslation(_groqKey))
                        results.add(apiTestManager.testGroqWhisper(_groqKey))
                    }
                    "deepl" -> results.add(apiTestManager.testDeepL(_deeplKey))
                    "claude" -> results.add(apiTestManager.testClaudeTranslation(_claudeKey))
                    "edge" -> results.add(apiTestManager.testEdgeTts())
                    "google_tts" -> results.add(apiTestManager.testGoogleTts())
                    "volcano" -> results.add(apiTestManager.testVolcanoTts(_volcanoAppId, _volcanoToken, _volcanoCluster))
                    "local" -> results.add(apiTestManager.testLocalServer(_localServerUrl, _localServerModel))
                }
                // Replace results for this type, keep others
                val oldNames = results.map { it.apiName }.toSet()
                _apiTestResults.removeAll { it.apiName in oldNames }
                _apiTestResults.addAll(0, results)
            } catch (e: Throwable) {
                log("API测试异常: ${e.message}")
            }
            _apiTestRunning = false
            _apiTestProgress = "完成"
        }
    }

    @Composable
    private fun VoskModelPanel() {
        val hasVosk = sherpaWhisperAsr.isVoskModelPresent()
        val sizeMB = if (hasVosk) sherpaWhisperAsr.voskModelSizeMB() else 0
        Column(Modifier.padding(start = 4.dp, top = 6.dp)) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    if (hasVosk) {
                        Text("Vosk 模型已就绪", fontSize = 13.sp, color = Color(0xFF4CAF50))
                        Text("占用 ${sizeMB}MB 存储空间", fontSize = 10.sp,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.45f))
                    } else {
                        Text("Vosk 模型未提取", fontSize = 13.sp, color = Color(0xFFEF5350))
                        Text("切换到 Vosk 录音时自动从 assets 提取", fontSize = 10.sp,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.45f))
                    }
                }
                if (hasVosk) {
                    TextButton(onClick = {
                        if (_recording && _asrEngine == 1) {
                            log("请先停止录音再删除模型")
                        } else {
                            sherpaWhisperAsr.deleteVoskModel()
                            _voskReady = false
                            voskModel?.close(); voskModel = null
                            log("已删除 Vosk 模型，释放 ${sizeMB}MB")
                        }
                    }) {
                        Text(S("delete"), fontSize = 12.sp, color = Color(0xFFEF5350))
                    }
                }
            }
        }
    }

    @Composable
    private fun WhisperModelSelector() {
        Column(Modifier.padding(start = 4.dp, top = 6.dp)) {
            Text(S("model_select"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
            Spacer(Modifier.height(4.dp))

            SherpaWhisperAsr.WhisperModel.entries.forEachIndexed { idx, m ->
                val dl = sherpaWhisperAsr.isModelDownloaded(m)
                val sizeMB = if (dl) sherpaWhisperAsr.modelSizeMB(m) else 0
                Row(
                    Modifier.fillMaxWidth().clip(RoundedCornerShape(6.dp))
                        .clickable { switchWhisperModel(idx) }
                        .background(if (_whisperModelIdx == idx) MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.35f) else Color.Transparent)
                        .padding(horizontal = 6.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    RadioButton(_whisperModelIdx == idx, { switchWhisperModel(idx) }, Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Column(Modifier.weight(1f)) {
                        Text(m.label, fontSize = 13.sp, fontWeight = if (_whisperModelIdx == idx) FontWeight.SemiBold else FontWeight.Normal)
                        Text("${m.desc} · ~${m.approxSizeMB}MB" +
                            if (dl) " · 已下载 (${sizeMB}MB)" else "",
                            fontSize = 10.sp, color = if (dl) Color(0xFF4CAF50) else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.45f))
                    }
                    if (dl) {
                        IconButton(onClick = {
                            if (_recording && _asrEngine == 4 && _whisperModelIdx == idx) {
                                log("请先停止录音再删除模型")
                            } else {
                                sherpaWhisperAsr.deleteModel(m)
                                if (_whisperModelIdx == idx) _localWhisperReady = false
                                log("已删除 Whisper ${m.label}")
                            }
                        }, modifier = Modifier.size(28.dp)) {
                            Icon(Icons.Default.Delete, S("delete"), Modifier.size(16.dp), tint = Color(0xFFEF5350))
                        }
                    }
                }
            }

            Spacer(Modifier.height(6.dp))
            val cur = selectedWhisperModel()
            if (_localWhisperReady) {
                Text("${cur.label} 已就绪", fontSize = 13.sp, color = Color(0xFF4CAF50))
            } else if (_localWhisperDownloading) {
                Text(_downloadProgress.ifBlank { "下载中…" }, fontSize = 13.sp, color = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.height(4.dp)); LinearProgressIndicator(Modifier.fillMaxWidth())
            } else {
                Button(onClick = { downloadWhisperModel() }, contentPadding = PaddingValues(horizontal = 14.dp, vertical = 4.dp)) {
                    Icon(Icons.Default.Download, null, Modifier.size(14.dp)); Spacer(Modifier.width(4.dp))
                    Text(if (sherpaWhisperAsr.isModelDownloaded(cur)) "重新初始化" else "下载 ${cur.label} (~${cur.approxSizeMB}MB)", fontSize = 12.sp)
                }
            }
        }
    }

    @Composable
    private fun StreamingAsrModelPanel() {
        Column(Modifier.padding(start = 4.dp, top = 6.dp)) {
            Text(S("model_select"), fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
            Spacer(Modifier.height(4.dp))

            SherpaStreamingAsr.StreamingModel.entries.forEachIndexed { idx, m ->
                val dl = streamingAsr.isModelDownloaded(m)
                val sizeMB = if (dl) streamingAsr.modelSizeMB(m) else 0
                Row(
                    Modifier.fillMaxWidth().clip(RoundedCornerShape(6.dp))
                        .clickable { switchStreamingAsrModel(idx) }
                        .background(if (_streamingAsrModelIdx == idx) MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.35f) else Color.Transparent)
                        .padding(horizontal = 6.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    RadioButton(_streamingAsrModelIdx == idx, { switchStreamingAsrModel(idx) }, Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Column(Modifier.weight(1f)) {
                        Text(m.label, fontSize = 13.sp, fontWeight = if (_streamingAsrModelIdx == idx) FontWeight.SemiBold else FontWeight.Normal)
                        Text("${m.desc}" + if (dl) " · 已下载 (${sizeMB}MB)" else "",
                            fontSize = 10.sp, color = if (dl) Color(0xFF4CAF50) else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.45f))
                    }
                    if (dl) {
                        IconButton(onClick = {
                            if (_recording && _asrEngine == 6 && _streamingAsrModelIdx == idx) {
                                log("请先停止录音再删除模型")
                            } else {
                                streamingAsr.deleteModel(m)
                                if (_streamingAsrModelIdx == idx) _streamingAsrReady = false
                                log("已删除流式ASR ${m.label}")
                            }
                        }, modifier = Modifier.size(28.dp)) {
                            Icon(Icons.Default.Delete, S("delete"), Modifier.size(16.dp), tint = Color(0xFFEF5350))
                        }
                    }
                }
            }

            Spacer(Modifier.height(6.dp))
            val cur = selectedStreamingModel()
            when {
                _streamingAsrReady ->
                    Text("${cur.label} 已就绪", fontSize = 13.sp, color = Color(0xFF4CAF50))
                _streamingAsrDownloading -> {
                    Text(_streamingAsrProgress.ifBlank { "下载中…" }, fontSize = 13.sp, color = MaterialTheme.colorScheme.primary)
                    Spacer(Modifier.height(4.dp)); LinearProgressIndicator(Modifier.fillMaxWidth())
                }
                else ->
                    Button(onClick = { downloadStreamingAsrModel() }, contentPadding = PaddingValues(horizontal = 14.dp, vertical = 4.dp)) {
                        Icon(Icons.Default.Download, null, Modifier.size(14.dp)); Spacer(Modifier.width(4.dp))
                        Text(if (streamingAsr.isModelDownloaded(cur)) "重新初始化" else "下载 ${cur.label} (~${cur.approxSizeMB}MB)", fontSize = 12.sp)
                    }
            }
            Text("Bilingual · No VAD · Realtime", fontSize = 10.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
                modifier = Modifier.padding(top = 4.dp))
        }
    }

    private var _sherpaTtsSid by mutableStateOf(0)

    @Composable
    private fun KokoroTtsPanel() {
        Column(Modifier.padding(start = 4.dp, top = 6.dp)) {
            val groups = VitsTts.Model.entries.groupBy { it.lang }
            val langLabels = mapOf("zh" to S("zh_models"), "en" to S("en_models"), "de" to S("other_models"))

            for ((lang, models) in groups) {
                Text(langLabels[lang] ?: lang, fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))

                models.forEach { model ->
                    val idx = model.ordinal
                    val downloaded = vitsTts.isModelDownloaded(model)
                    val active = _zhTtsModelIdx == idx && _vitsTtsReady && vitsTts.currentModelLabel() == model.label
                    Row(Modifier.fillMaxWidth().clickable {
                        _zhTtsModelIdx = idx; saveInt("zh_tts_model_idx", idx)
                        if (downloaded) switchSherpaModel(model)
                    }.padding(vertical = 3.dp), verticalAlignment = Alignment.CenterVertically) {
                        RadioButton(selected = _zhTtsModelIdx == idx, onClick = null, Modifier.size(18.dp))
                        Spacer(Modifier.width(4.dp))
                        Column(Modifier.weight(1f)) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(model.label, fontSize = 12.sp,
                                    color = if (active) Color(0xFF4CAF50) else MaterialTheme.colorScheme.onSurface)
                                if (active) Text(" ●", fontSize = 8.sp, color = Color(0xFF4CAF50))
                            }
                            Text(buildString {
                                append(model.engineType.name)
                                if (model.speakers > 1) append(" · ${model.speakers}音色")
                                append(" · ~${model.approxSizeMB}MB")
                                if (downloaded) append(" · ✓")
                            }, fontSize = 9.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.45f))
                        }
                        // Download / Delete buttons
                        if (!downloaded && !model.isBuiltin) {
                            TextButton(onClick = { downloadZhTtsModel(model) },
                                contentPadding = PaddingValues(horizontal = 4.dp, vertical = 0.dp)) {
                                Text("下载", fontSize = 10.sp)
                            }
                        } else if (downloaded && !model.isBuiltin) {
                            TextButton(onClick = {
                                vitsTts.deleteModel(model)
                                if (active) _vitsTtsReady = false
                                log(S("delete"))
                            }, contentPadding = PaddingValues(horizontal = 4.dp, vertical = 0.dp)) {
                                Text(S("delete"), fontSize = 10.sp, color = Color(0xFFEF5350))
                            }
                        }
                    }
                }
                Spacer(Modifier.height(4.dp))
            }

            if (_zhTtsDownloading) {
                Text(_zhTtsProgress.ifBlank { "下载中…" }, fontSize = 11.sp, color = MaterialTheme.colorScheme.primary)
                LinearProgressIndicator(Modifier.fillMaxWidth().padding(top = 2.dp))
            }

            // Speaker ID (for multi-speaker models)
            val curModel = selectedZhTtsModel()
            if (_vitsTtsReady && curModel.speakers > 1) {
                Spacer(Modifier.height(8.dp))
                Text("音色 (Speaker ID: $_sherpaTtsSid / ${curModel.speakers - 1})", fontSize = 11.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                Slider(
                    value = _sherpaTtsSid.toFloat(),
                    onValueChange = { _sherpaTtsSid = it.toInt(); saveInt("sherpa_tts_sid", _sherpaTtsSid) },
                    valueRange = 0f..(curModel.speakers - 1).toFloat(),
                    steps = (curModel.speakers - 2).coerceAtLeast(0),
                    modifier = Modifier.fillMaxWidth()
                )
            }


            Text(S("offline_tts_desc"), fontSize = 9.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.35f),
                modifier = Modifier.padding(top = 4.dp))
        }
    }

    @Composable
    private fun VolcanoTtsPanel() {
        Column(Modifier.padding(start = 12.dp, top = 6.dp)) {
            Text(
                S("volcano_panel_title"),
                fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
            )
            Text(
                S("volcano_panel_desc"),
                fontSize = 10.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.45f),
                modifier = Modifier.padding(top = 2.dp, bottom = 6.dp)
            )

            OutlinedTextField(
                value = _volcanoAppId,
                onValueChange = {
                    _volcanoAppId = it.trim()
                    saveKey("volcano_app_id", _volcanoAppId)
                },
                label = { Text("App ID", fontSize = 11.sp) },
                singleLine = true,
                textStyle = androidx.compose.ui.text.TextStyle(fontSize = 12.sp),
                modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp)
            )
            OutlinedTextField(
                value = _volcanoToken,
                onValueChange = {
                    _volcanoToken = it.trim()
                    saveKey("volcano_access_token", _volcanoToken)
                },
                label = { Text("Access Token", fontSize = 11.sp) },
                singleLine = true,
                visualTransformation = PasswordVisualTransformation(),
                textStyle = androidx.compose.ui.text.TextStyle(fontSize = 12.sp),
                modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp)
            )
            OutlinedTextField(
                value = _volcanoCluster,
                onValueChange = {
                    _volcanoCluster = it.trim().ifBlank { com.example.myapplication1.tts.VolcanoTts.DEFAULT_CLUSTER }
                    saveKey("volcano_cluster", _volcanoCluster)
                },
                label = { Text("Cluster (默认 volcano_tts)", fontSize = 11.sp) },
                singleLine = true,
                textStyle = androidx.compose.ui.text.TextStyle(fontSize = 12.sp),
                modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp)
            )

            Spacer(Modifier.height(6.dp))
            val voices = com.example.myapplication1.tts.VolcanoTts.voicesForLang(_targetLang)
            Text(
                "语音角色 (${_targetLang.uppercase()})",
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
            )
            voices.forEachIndexed { i, (_, label) ->
                SmallRadio(label, _volcanoVoiceIdx == i) {
                    _volcanoVoiceIdx = i
                    saveInt("volcano_voice_idx", i)
                    log("火山语音: $label")
                }
            }
            if (_volcanoAppId.isBlank() || _volcanoToken.isBlank()) {
                Text(
                    S("volcano_need_creds"),
                    fontSize = 11.sp, color = Color(0xFFEF5350),
                    modifier = Modifier.padding(start = 4.dp, top = 4.dp)
                )
            }
        }
    }

    private fun switchSherpaModel(model: VitsTts.Model) {
        _vitsTtsReady = false
        lifecycleScope.launch(Dispatchers.IO) {
            val ok = vitsTts.initModel(model)
            withContext(Dispatchers.Main) {
                _vitsTtsReady = ok
                log(if (ok) "${model.label} 就绪" else "${model.label} 初始化失败")
            }
        }
    }

    private fun downloadZhTtsModel(model: VitsTts.Model) {
        if (_zhTtsDownloading) return
        _zhTtsDownloading = true
        lifecycleScope.launch {
            val cb = object : VitsTts.Callback {
                override fun onDownloadProgress(file: String, percent: Int) { _zhTtsProgress = "$file: $percent%" }
                override fun onError(message: String) { log("下载: $message") }
            }
            if (vitsTts.downloadModel(model, cb)) {
                val ok = withContext(Dispatchers.IO) { vitsTts.initModel(model) }
                _vitsTtsReady = ok
                log(if (ok) "${model.label} 就绪" else "${model.label} 初始化失败")
            }
            _zhTtsDownloading = false; _zhTtsProgress = ""
        }
    }

    @Composable
    private fun OfflineTransModelPanel() {
        val model = if (_translationEngineType == 6) OnDeviceTranslationModel.NLLB_600M_INT8
                    else OnDeviceTranslationModel.OPUS_MT_EN_ZH
        val downloaded = model.isDownloaded(this@MainActivity)

        Column(Modifier.padding(start = 4.dp, top = 6.dp)) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(model.label, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                    Text("${model.desc} · ~${model.approxSizeMB}MB" +
                        if (downloaded) " · 已下载" else "",
                        fontSize = 10.sp,
                        color = if (downloaded) Color(0xFF4CAF50)
                               else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.45f))
                }
            }
            Spacer(Modifier.height(6.dp))

            if (downloaded) {
                Text("模型已就绪", fontSize = 13.sp, color = Color(0xFF4CAF50))
                Spacer(Modifier.height(4.dp))
                TextButton(onClick = {
                    translationModelManager.deleteModel(model)
                    invalidateTranslationCache()
                    log(S("delete"))
                }) {
                    Text("删除模型", fontSize = 12.sp, color = Color(0xFFEF5350))
                }
            } else if (_offlineTransDownloading) {
                Text(_offlineTransProgress.ifBlank { "下载中…" }, fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.height(4.dp))
                LinearProgressIndicator(Modifier.fillMaxWidth())
            } else {
                Button(onClick = { downloadOfflineTransModel(model) },
                    contentPadding = PaddingValues(horizontal = 14.dp, vertical = 4.dp)) {
                    Icon(Icons.Default.Download, null, Modifier.size(14.dp))
                    Spacer(Modifier.width(4.dp))
                    Text("下载 ${model.label} (~${model.approxSizeMB}MB)", fontSize = 12.sp)
                }
            }
            Text("使用 ONNX Runtime 在设备上运行翻译模型，无需网络",
                fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
                modifier = Modifier.padding(top = 4.dp))
        }
    }

    private fun downloadOfflineTransModel(model: OnDeviceTranslationModel) {
        if (_offlineTransDownloading) return
        _offlineTransDownloading = true
        val cb = object : TranslationModelManager.DownloadCallback {
            override fun onProgress(file: String, percent: Int) {
                runOnUiThread { _offlineTransProgress = "$file: $percent%" }
            }
            override fun onComplete(success: Boolean, error: String?) {
                runOnUiThread {
                    _offlineTransDownloading = false
                    _offlineTransProgress = ""
                    if (success) {
                        invalidateTranslationCache()
                        log("${model.label} 下载完成")
                    } else {
                        log("下载失败: $error")
                    }
                }
            }
        }
        lifecycleScope.launch { translationModelManager.download(model, cb) }
    }

    // ---- History ----

    @Composable
    private fun HistoryArea(modifier: Modifier = Modifier) {
        val sdf = remember { SimpleDateFormat("MM-dd HH:mm", Locale.getDefault()) }
        val ls = rememberLazyListState()
        val filtered = remember(_historySessions.size, _historySearchQuery) {
            if (_historySearchQuery.isBlank()) _historySessions.toList()
            else translationHistory.search(_historySearchQuery)
        }
        // 会话序号映射：按创建时间升序编号
        val numberMap = remember(_historySessions.size) {
            translationHistory.allSessions().sortedBy { it.startTime }
                .withIndex().associate { (i, s) -> s.id to (i + 1) }
        }
        val currentId = translationHistory.currentSession()?.id

        // Rename dialog
        if (_editingSessionId != null) {
            AlertDialog(
                onDismissRequest = { _editingSessionId = null },
                title = { Text("重命名会话") },
                text = {
                    OutlinedTextField(_editingTitle, { _editingTitle = it },
                        label = { Text("标题") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                },
                confirmButton = {
                    TextButton({
                        translationHistory.renameSession(_editingSessionId!!, _editingTitle)
                        _historySessions.clear(); _historySessions.addAll(translationHistory.allSessions())
                        _editingSessionId = null
                    }) { Text(S("save")) }
                },
                dismissButton = { TextButton({ _editingSessionId = null }) { Text(S("cancel")) } }
            )
        }

        // 单会话删除确认
        if (_pendingDeleteSessionId != null) {
            val target = _historySessions.find { it.id == _pendingDeleteSessionId }
            val isCurrentTarget = _pendingDeleteSessionId == currentId
            AlertDialog(
                onDismissRequest = { _pendingDeleteSessionId = null },
                icon = { Icon(Icons.Default.Warning, null, tint = Color(0xFFEF5350)) },
                title = { Text("删除对话？") },
                text = {
                    Column {
                        Text("\"${target?.displayTitle ?: ""}\" 将被永久删除（${target?.entries?.size ?: 0} 条记录），此操作无法撤销。")
                        if (isCurrentTarget) {
                            Spacer(Modifier.height(8.dp))
                            Text("⚠ 这是当前对话，删除后主界面将被清空。",
                                 fontSize = 12.sp, color = Color(0xFFEF5350))
                        }
                    }
                },
                confirmButton = {
                    TextButton(onClick = {
                        _pendingDeleteSessionId?.let { handleDeleteSession(it) }
                        _pendingDeleteSessionId = null
                    }) { Text(S("delete"), color = Color(0xFFEF5350)) }
                },
                dismissButton = {
                    TextButton({ _pendingDeleteSessionId = null }) { Text(S("cancel")) }
                }
            )
        }

        // 清空全部确认
        if (_showClearAllDialog) {
            AlertDialog(
                onDismissRequest = { _showClearAllDialog = false },
                icon = { Icon(Icons.Default.Warning, null, tint = Color(0xFFEF5350)) },
                title = { Text("清空所有历史？") },
                text = { Text("将删除所有 ${_historySessions.size} 个对话，此操作无法撤销。") },
                confirmButton = {
                    TextButton(onClick = {
                        handleClearAllSessions()
                        _showClearAllDialog = false
                    }) { Text(S("clear_background"), color = Color(0xFFEF5350)) }
                },
                dismissButton = {
                    TextButton({ _showClearAllDialog = false }) { Text(S("cancel")) }
                }
            )
        }

        Column(modifier.fillMaxWidth()) {
            // 计数（新建/清空按钮已移至 HistoryScreen 的 TopAppBar）
            Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("共 ${filtered.size} 个会话", fontSize = 12.sp,
                     color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
            }
            // Search bar
            OutlinedTextField(
                _historySearchQuery,
                { _historySearchQuery = it },
                placeholder = { Text("搜索标题或内容…", fontSize = 13.sp) },
                leadingIcon = { Icon(Icons.Default.Search, null, Modifier.size(18.dp)) },
                trailingIcon = {
                    if (_historySearchQuery.isNotBlank()) {
                        IconButton({ _historySearchQuery = "" }) { Icon(Icons.Default.Close, null, Modifier.size(16.dp)) }
                    }
                },
                singleLine = true,
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp).height(48.dp),
                textStyle = LocalTextStyle.current.copy(fontSize = 13.sp)
            )
            Spacer(Modifier.height(6.dp))

            if (filtered.isEmpty()) {
                Box(Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                    Text(
                        if (_historySearchQuery.isNotBlank()) "未找到匹配记录" else "暂无翻译记录",
                        fontSize = 14.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
                    )
                }
            } else {
                LazyColumn(state = ls, modifier = Modifier.fillMaxWidth().weight(1f).padding(horizontal = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp), contentPadding = PaddingValues(vertical = 8.dp),
                    reverseLayout = true) {
                    items(filtered.size) { i ->
                        val idx = filtered.size - 1 - i
                        val session = filtered[idx]
                        val num = numberMap[session.id] ?: 0
                        val isCurrent = session.id == currentId
                        Card(
                            onClick = { loadSessionIntoMain(session.id) },
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(12.dp),
                            colors = CardDefaults.cardColors(
                                containerColor = if (isCurrent)
                                    MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.18f)
                                else MaterialTheme.colorScheme.surface
                            ),
                            elevation = CardDefaults.cardElevation(1.dp)
                        ) {
                            Column(Modifier.padding(14.dp)) {
                                // Title row with number badge + actions
                                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                                    // 序号徽章
                                    Surface(
                                        shape = CircleShape,
                                        color = if (isCurrent) MaterialTheme.colorScheme.primary
                                                else MaterialTheme.colorScheme.surfaceVariant,
                                        modifier = Modifier.size(28.dp)
                                    ) {
                                        Box(contentAlignment = Alignment.Center) {
                                            Text("#$num", fontSize = 11.sp, fontWeight = FontWeight.Bold,
                                                 color = if (isCurrent) Color.White
                                                         else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                                        }
                                    }
                                    Spacer(Modifier.width(8.dp))
                                    Text(session.displayTitle, fontSize = 14.sp, fontWeight = FontWeight.SemiBold,
                                        modifier = Modifier.weight(1f, fill = false), maxLines = 1, overflow = TextOverflow.Ellipsis)
                                    if (isCurrent) {
                                        Spacer(Modifier.width(6.dp))
                                        Surface(shape = RoundedCornerShape(4.dp),
                                                color = MaterialTheme.colorScheme.primary) {
                                            Text("当前", fontSize = 9.sp, color = Color.White,
                                                 modifier = Modifier.padding(horizontal = 4.dp, vertical = 1.dp))
                                        }
                                    }
                                    Spacer(Modifier.weight(1f))
                                    // Rename
                                    IconButton(onClick = {
                                        _editingSessionId = session.id
                                        _editingTitle = session.title
                                    }, modifier = Modifier.size(28.dp)) {
                                        Icon(Icons.Default.Edit, S("rename"), Modifier.size(14.dp),
                                            tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f))
                                    }
                                    // Delete → 确认对话框
                                    IconButton(onClick = { _pendingDeleteSessionId = session.id },
                                               modifier = Modifier.size(28.dp)) {
                                        Icon(Icons.Default.Delete, S("delete"), Modifier.size(14.dp),
                                            tint = Color(0xFFEF5350).copy(alpha = 0.7f))
                                    }
                                }
                                // Time + count
                                Row(Modifier.fillMaxWidth()) {
                                    Text(sdf.format(Date(session.startTime)), fontSize = 10.sp,
                                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f))
                                    Spacer(Modifier.weight(1f))
                                    Text("${session.entries.size} 句", fontSize = 10.sp,
                                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f))
                                }
                                Spacer(Modifier.height(6.dp))
                                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.3f))
                                Spacer(Modifier.height(6.dp))

                                // 前两条 entry 预览
                                val entries = session.entries
                                if (entries.isEmpty()) {
                                    Text("（无内容）", fontSize = 12.sp, fontStyle = FontStyle.Italic,
                                         color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f))
                                } else {
                                    entries.take(2).forEachIndexed { entryIdx, entry ->
                                        Row(Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
                                            Text(
                                                "${entryIdx + 1}.",
                                                fontSize = 11.sp, fontWeight = FontWeight.SemiBold,
                                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.35f),
                                                modifier = Modifier.width(22.dp)
                                            )
                                            Column(Modifier.weight(1f)) {
                                                Text(
                                                    entry.en, fontSize = 12.sp, lineHeight = 16.sp,
                                                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                                                    maxLines = 2, overflow = TextOverflow.Ellipsis
                                                )
                                                if (entry.zh.isNotBlank()) {
                                                    Text(
                                                        entry.zh, fontSize = 13.sp, lineHeight = 18.sp,
                                                        fontWeight = FontWeight.Medium,
                                                        maxLines = 2, overflow = TextOverflow.Ellipsis,
                                                        modifier = Modifier.padding(top = 1.dp)
                                                    )
                                                }
                                            }
                                        }
                                    }
                                    // 点击继续提示
                                    if (entries.size > 2) {
                                        Text(
                                            "… ${entries.size} ${S("entries")}",
                                            fontSize = 10.sp, fontStyle = FontStyle.Italic,
                                            color = MaterialTheme.colorScheme.primary.copy(alpha = 0.7f),
                                            modifier = Modifier.padding(start = 22.dp, top = 2.dp)
                                        )
                                    } else {
                                        Text(
                                            "Tap to continue",
                                            fontSize = 10.sp, fontStyle = FontStyle.Italic,
                                            color = MaterialTheme.colorScheme.primary.copy(alpha = 0.7f),
                                            modifier = Modifier.padding(start = 22.dp, top = 2.dp)
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- Language Bar (Iter-4) ----

    @Composable
    private fun LanguageBar() {
        val langNames = mapOf(
            "en" to "English", "zh" to "中文", "ja" to "日本語", "ko" to "한국어",
            "fr" to "Français", "de" to "Deutsch", "es" to "Español", "ru" to "Русский",
            "auto" to "自动"
        )
        var showSourcePicker by remember { mutableStateOf(false) }
        var showTargetPicker by remember { mutableStateOf(false) }
        val allLangs = listOf("en", "zh", "ja", "ko", "fr", "de", "es", "ru")

        Surface(color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f), modifier = Modifier.fillMaxWidth()) {
            Row(
                Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center,
            ) {
                // Source language button
                Surface(
                    onClick = { showSourcePicker = true },
                    shape = RoundedCornerShape(8.dp),
                    color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.6f),
                ) {
                    Text(
                        if (_langAutoMode && _detectedLang.isNotBlank()) "${langNames[_detectedLang] ?: _detectedLang} (自动)"
                        else langNames[_sourceLang] ?: _sourceLang,
                        fontSize = 13.sp, fontWeight = FontWeight.Medium,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                    )
                }

                Spacer(Modifier.width(8.dp))
                // Swap button
                IconButton(onClick = {
                    val tmp = _sourceLang; _sourceLang = _targetLang; _targetLang = tmp
                    saveKey("source_lang", _sourceLang); saveKey("target_lang", _targetLang)
                    updatePipelineContext(); translationPipeline.clearCache()
                    autoSelectTranslationEngine()
                    log("翻译方向: ${_sourceLang.uppercase()}→${_targetLang.uppercase()}")
                }, modifier = Modifier.size(28.dp)) {
                    Icon(Icons.Default.SwapHoriz, "切换", Modifier.size(20.dp), tint = MaterialTheme.colorScheme.primary)
                }
                Spacer(Modifier.width(8.dp))

                // Target language button
                Surface(
                    onClick = { showTargetPicker = true },
                    shape = RoundedCornerShape(8.dp),
                    color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.6f),
                ) {
                    Text(
                        langNames[_targetLang] ?: _targetLang,
                        fontSize = 13.sp, fontWeight = FontWeight.Medium,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                    )
                }

                // Auto indicator
                if (_langAutoMode && _lidReady) {
                    Spacer(Modifier.width(6.dp))
                    Text("自动", fontSize = 9.sp, color = MaterialTheme.colorScheme.primary.copy(alpha = 0.7f))
                }
            }
        }

        // Source language picker dropdown
        if (showSourcePicker) {
            AlertDialog(
                onDismissRequest = { showSourcePicker = false },
                title = { Text(S("source_label"), fontSize = 16.sp) },
                text = {
                    Column {
                        allLangs.forEach { lang ->
                            TextButton(onClick = {
                                _sourceLang = lang; saveKey("source_lang", lang)
                                _langAutoMode = false; saveBool("lang_auto_mode", false)
                                streamingAsr.languageDetectionEnabled = false
                                updatePipelineContext(); translationPipeline.clearCache()
                                autoSelectTranslationEngine()
                                showSourcePicker = false
                            }, modifier = Modifier.fillMaxWidth()) {
                                Text("${langNames[lang]} (${lang.uppercase()})", fontSize = 14.sp,
                                    modifier = Modifier.fillMaxWidth(),
                                    fontWeight = if (lang == _sourceLang) FontWeight.Bold else FontWeight.Normal)
                            }
                        }
                    }
                },
                confirmButton = { TextButton({ showSourcePicker = false }) { Text(S("cancel")) } },
            )
        }

        // Target language picker dropdown
        if (showTargetPicker) {
            AlertDialog(
                onDismissRequest = { showTargetPicker = false },
                title = { Text(S("target_label"), fontSize = 16.sp) },
                text = {
                    Column {
                        allLangs.forEach { lang ->
                            TextButton(onClick = {
                                _targetLang = lang; saveKey("target_lang", lang)
                                updatePipelineContext(); translationPipeline.clearCache()
                                autoSelectTranslationEngine()
                                showTargetPicker = false
                            }, modifier = Modifier.fillMaxWidth()) {
                                Text("${langNames[lang]} (${lang.uppercase()})", fontSize = 14.sp,
                                    modifier = Modifier.fillMaxWidth(),
                                    fontWeight = if (lang == _targetLang) FontWeight.Bold else FontWeight.Normal)
                            }
                        }
                    }
                },
                confirmButton = { TextButton({ showTargetPicker = false }) { Text(S("cancel")) } },
            )
        }
    }

    // ---- Conversation ----

    @Composable
    private fun ConversationArea(modifier: Modifier = Modifier) {
        val ls = rememberLazyListState()
        val paras = _paragraphs  // reading mutableStateOf triggers recomposition on any write
        val totalItems = paras.size + (if (_currentPartial.isNotBlank()) 1 else 0)
        // Instantly scroll to the newest item whenever the list size changes
        // Instantly scroll to the newest item whenever the list changes (including content updates)
        LaunchedEffect(paras, _currentPartial) {
            if (totalItems > 0) {
                ls.scrollToItem(totalItems - 1)
            }
        }
        LazyColumn(
            state = ls,
            modifier = modifier
                .padding(horizontal = 16.dp)
            .padding(bottom = 80.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
            contentPadding = PaddingValues(vertical = 12.dp)
        ) {
            items(items = paras, key = { it.id }) { para ->
                ParagraphCard(para)
            }
            if (_currentPartial.isNotBlank() && (_recording || _mediaCaptureActive)) {
                item { PartialCard(_currentPartial) }
            }
            if (_paragraphs.isEmpty() && _currentPartial.isBlank()) item { EmptyHint() }
        }
    }

    @Composable
    private fun ParagraphCard(para: Paragraph) {
        Card(Modifier.fillMaxWidth().animateContentSize(), shape = RoundedCornerShape(14.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            elevation = CardDefaults.cardElevation(1.dp)
        ) {
            Column(Modifier.padding(14.dp)) {
                // Source text (ASR) — always on top, with language label
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(_sourceLang.uppercase(), fontSize = 9.sp, fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary.copy(alpha = 0.5f),
                        modifier = Modifier.padding(end = 6.dp))
                    Text(para.combinedEn, fontSize = 14.sp, lineHeight = 20.sp,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                        modifier = Modifier.weight(1f))
                }

                if (_enableTranslation) {
                    Spacer(Modifier.height(6.dp))
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.3f))
                    Spacer(Modifier.height(6.dp))

                    // Target text (translation) — always below source
                    val zh = para.rawZh
                    if (zh.isNotBlank()) {
                        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                            Text(_targetLang.uppercase(), fontSize = 9.sp, fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.secondary.copy(alpha = 0.5f),
                                modifier = Modifier.padding(end = 6.dp))
                            Text(zh, fontSize = 17.sp, fontWeight = FontWeight.Medium, lineHeight = 24.sp,
                                modifier = Modifier.weight(1f))
                            if (zh != "[翻译失败]" && zh != "[Translation failed]") {
                                IconButton({ speakManual(zh) }, Modifier.size(32.dp)) {
                                    Icon(Icons.AutoMirrored.Filled.VolumeUp, S("play"),
                                        Modifier.size(18.dp), tint = MaterialTheme.colorScheme.primary)
                                }
                            }
                        }
                    } else if (para.anyTranslating) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp)
                            Spacer(Modifier.width(8.dp))
                            Text(S("translating"), fontSize = 13.sp, color = MaterialTheme.colorScheme.primary)
                        }
                    }

                    // Status: translation progress or quality upgrade indicator
                    val hasUpgrade = para.segments.any { it.qualityZh.isNotBlank() }
                    if (zh.isNotBlank() && para.anyTranslating) {
                        Spacer(Modifier.height(4.dp))
                        val done = para.segments.count { !it.translating }
                        Text(S("translating"), fontSize = 10.sp,
                            color = MaterialTheme.colorScheme.primary.copy(alpha = 0.6f))
                    } else if (hasUpgrade) {
                        Spacer(Modifier.height(4.dp))
                        Text(S("optimized"), fontSize = 10.sp, color = Color(0xFF4CAF50).copy(alpha = 0.7f))
                    }
                }
            }
        }
    }

    @Composable
    private fun PartialCard(text: String) {
        Card(Modifier.fillMaxWidth(), shape = RoundedCornerShape(14.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f))
        ) {
            Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
                val p by rememberInfiniteTransition().animateFloat(0.8f, 1.2f,
                    infiniteRepeatable(tween(600), RepeatMode.Reverse))
                Box(Modifier.size(8.dp).scale(p).clip(CircleShape).background(Color(0xFFEF5350)))
                Spacer(Modifier.width(10.dp))
                Text(text, fontSize = 15.sp, fontStyle = FontStyle.Italic,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
            }
        }
    }

    @Composable
    private fun EmptyHint() {
        Box(Modifier.fillMaxWidth().padding(vertical = 60.dp), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(Icons.Default.Mic, null, Modifier.size(48.dp), tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.15f))
                Spacer(Modifier.height(12.dp))
                Text("点击下方按钮开始录音", fontSize = 15.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.35f))
                Text("说英语，自动翻译成中文", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.25f))
            }
        }
    }

    @Composable
    private fun RecordFAB(recording: Boolean, onClick: () -> Unit) {
        val isMediaCapture = _mediaCaptureActive
        val bg by animateColorAsState(
            if (isMediaCapture) Color(0xFF4CAF50)
            else if (recording) Color(0xFFEF5350) else Color(0xFF5B5FC7)
        )
        val p by rememberInfiniteTransition().animateFloat(1f, if (recording || isMediaCapture) 1.08f else 1f, infiniteRepeatable(tween(800), RepeatMode.Reverse))
        val label = when {
            isMediaCapture -> ""
            recording -> ""
            else -> "点击开始录音"
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(label, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
            Spacer(Modifier.height(6.dp))
            FloatingActionButton(onClick, Modifier.size(64.dp).scale(p).shadow(if (recording || isMediaCapture) 12.dp else 6.dp, CircleShape), containerColor = bg, contentColor = Color.White, shape = CircleShape) {
                Icon(
                    if (isMediaCapture) Icons.Default.Audiotrack
                    else if (recording) Icons.Default.Stop else Icons.Default.Mic,
                    label, Modifier.size(28.dp)
                )
            }
        }
    }
}

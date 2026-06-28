package com.example.myapplication1

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.*
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.IBinder
import android.util.Log
import kotlinx.coroutines.*
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import java.io.File
import java.io.FileOutputStream

/**
 * Foreground service that captures system/app audio via MediaProjection
 * and runs Vosk ASR on the captured audio.
 */
class MediaCaptureService : Service() {

    companion object {
        private const val TAG = "MediaCaptureSvc"
        const val CHANNEL_ID = "media_capture_channel"
        const val NOTIFICATION_ID = 1001
        const val ACTION_START = "START_CAPTURE"
        const val ACTION_STOP = "STOP_CAPTURE"
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_RESULT_DATA = "result_data"

        @Volatile var isCapturing = false
        @Volatile var captureStatus = ""    // 状态文本
        @Volatile var sourceAppName = ""    // 音源应用名
        @Volatile var lastError = ""        // 上次错误

        var asrCallback: AsrCallback? = null

        interface AsrCallback {
            fun onPartial(text: String)
            fun onResult(text: String)
            fun onStateChanged(capturing: Boolean, status: String, sourceApp: String)
            fun onError(msg: String)
        }
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private var mediaProjection: MediaProjection? = null
    private var audioRecord: AudioRecord? = null
    private var asrJob: Job? = null
    private var sourceDetectJob: Job? = null
    private var voskModel: Model? = null
    private var captureSampleRate = 48000

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification("媒体转译准备中…"))
        loadVoskModel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, 0)
                val data = if (Build.VERSION.SDK_INT >= 33) {
                    intent.getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    intent.getParcelableExtra(EXTRA_RESULT_DATA)
                }
                if (data != null) {
                    startCapture(resultCode, data)
                } else {
                    reportError("媒体投影数据为空")
                    stopForeground(STOP_FOREGROUND_REMOVE)
                    stopSelf()
                }
            }
            ACTION_STOP -> {
                stopCapture()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        stopCapture()
        try { voskModel?.close() } catch (_: Throwable) {}
        scope.cancel()
        isCapturing = false
        captureStatus = ""
        sourceAppName = ""
        super.onDestroy()
    }

    // ===================== Vosk Model =====================

    private fun loadVoskModel() {
        scope.launch(Dispatchers.IO) {
            try {
                val dst = File(filesDir, "model-en")
                if (!dst.exists()) copyAssetDir("model-en", dst)
                voskModel = Model(dst.absolutePath)
                Log.i(TAG, "Vosk model loaded")
            } catch (e: Throwable) {
                Log.e(TAG, "Vosk load failed: ${e.message}")
            }
        }
    }

    // ===================== Capture =====================

    @SuppressLint("MissingPermission")
    private fun startCapture(resultCode: Int, data: Intent) {
        if (isCapturing) return
        if (Build.VERSION.SDK_INT < 29) {
            reportError("需要 Android 10+")
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return
        }

        try {
            val mgr = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            mediaProjection = mgr.getMediaProjection(resultCode, data)
            if (mediaProjection == null) {
                reportError("无法获取媒体投影")
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
                return
            }
        } catch (e: Throwable) {
            reportError("媒体投影失败: ${e.message}")
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return
        }

        // Build capture config — only USAGE_MEDIA and USAGE_GAME from other apps
        // TTS uses USAGE_ASSISTANT so it won't be captured; excludeUid is extra safety
        val captureConfig = AudioPlaybackCaptureConfiguration.Builder(mediaProjection!!)
            .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
            .addMatchingUsage(AudioAttributes.USAGE_GAME)
            .addMatchingUsage(AudioAttributes.USAGE_UNKNOWN)
            .excludeUid(android.os.Process.myUid())
            .build()

        // Try 48000 first (matches system mixer), then 16000
        audioRecord = tryCreateAudioRecord(captureConfig, 48000)
            ?: tryCreateAudioRecord(captureConfig, 16000)

        if (audioRecord == null) {
            reportError("无法创建音频录制，设备可能不支持媒体音频捕获")
            try { mediaProjection?.stop() } catch (_: Throwable) {}
            mediaProjection = null
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return
        }

        try {
            audioRecord?.startRecording()
        } catch (e: Throwable) {
            reportError("录制启动失败: ${e.message}")
            releaseAudioRecord()
            try { mediaProjection?.stop() } catch (_: Throwable) {}
            mediaProjection = null
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return
        }

        isCapturing = true
        lastError = ""
        updateStatus("媒体转译中")
        updateNotification("媒体转译中…")
        asrCallback?.onStateChanged(true, "媒体转译中", "")
        startAsrLoop()
        startSourceDetection()
    }

    private fun tryCreateAudioRecord(config: AudioPlaybackCaptureConfiguration, sampleRate: Int): AudioRecord? {
        return try {
            val fmt = AudioFormat.Builder()
                .setSampleRate(sampleRate)
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                .build()

            // Use a smaller buffer (≈0.5 s) to reduce latency
            // Use a larger buffer (≈1 s) to improve ASR stability
            val bufSize = sampleRate / 2 // ~0.25‑second buffer (sampleRate * 2 / 4 bytes)
            val record = AudioRecord.Builder()
                .setAudioPlaybackCaptureConfig(config)
                .setAudioFormat(fmt)
                .setBufferSizeInBytes(bufSize)
                .build()

            if (record.state == AudioRecord.STATE_INITIALIZED) {
                captureSampleRate = sampleRate
                Log.i(TAG, "AudioRecord created at ${sampleRate}Hz")
                record
            } else {
                try { record.release() } catch (_: Throwable) {}
                null
            }
        } catch (e: Throwable) {
            Log.w(TAG, "AudioRecord at ${sampleRate}Hz failed: ${e.message}")
            null
        }
    }

    private fun stopCapture() {
        asrJob?.cancel(); asrJob = null
        sourceDetectJob?.cancel(); sourceDetectJob = null
        releaseAudioRecord()
        try { mediaProjection?.stop() } catch (_: Throwable) {}
        mediaProjection = null
        isCapturing = false
        updateStatus("")
        sourceAppName = ""
        asrCallback?.onStateChanged(false, "", "")
    }

    private fun releaseAudioRecord() {
        try { audioRecord?.stop() } catch (_: Throwable) {}
        try { audioRecord?.release() } catch (_: Throwable) {}
        audioRecord = null
    }

    // ===================== ASR Loop =====================

    private fun startAsrLoop() {
        asrJob = scope.launch(Dispatchers.IO) {
            // Wait for Vosk model
            var waited = 0
            while (voskModel == null && waited < 10000) { delay(200); waited += 200 }
            if (voskModel == null) {
                withContext(Dispatchers.Main) { reportError("Vosk 模型未就绪") }
                return@launch
            }

            val recognizer = Recognizer(voskModel, 16000f)
            val needsResample = captureSampleRate != 16000
            val ratio = if (needsResample) captureSampleRate / 16000 else 1

            // Read buffer: must be multiple of ratio for clean resampling
            // Smaller read chunk for lower latency
            // Smaller read chunk for lower latency
            val readSize = if (needsResample) 512 * ratio else 512
            val readBuf = ShortArray(readSize)

            try {
                while (isActive && isCapturing) {
                    val n = audioRecord?.read(readBuf, 0, readBuf.size) ?: break
                    if (n <= 0) continue

                    // Resample if needed (48k -> 16k = 3:1 average)
                    val asrBuf: ShortArray
                    val asrLen: Int
                    if (needsResample) {
                        asrLen = n / ratio
                        asrBuf = ShortArray(asrLen)
                        for (i in 0 until asrLen) {
                            var sum = 0L
                            for (j in 0 until ratio) {
                                sum += readBuf[i * ratio + j]
                            }
                            asrBuf[i] = (sum / ratio).toInt().toShort()
                        }
                    } else {
                        asrBuf = readBuf
                        asrLen = n
                    }

                    try {
                        if (recognizer.acceptWaveForm(asrBuf, asrLen)) {
                            val t = JSONObject(recognizer.result).optString("text").trim()
                            if (t.isNotBlank()) {
                                withContext(Dispatchers.Main) { asrCallback?.onResult(t) }
                            }
                        } else {
                            val p = JSONObject(recognizer.partialResult).optString("partial").trim()
                            if (p.isNotBlank()) {
                                withContext(Dispatchers.Main) { asrCallback?.onPartial(p) }
                            }
                        }
                    } catch (_: Throwable) {}
                }
            } finally {
                recognizer.close()
            }
        }
    }

    // ===================== Source App Detection =====================

    @SuppressLint("NewApi")
    private fun startSourceDetection() {
        sourceDetectJob = scope.launch(Dispatchers.IO) {
            while (isActive && isCapturing) {
                try {
                    val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
                    val configs = am.activePlaybackConfigurations
                    val apps = configs.mapNotNull { config ->
                        try {
                            val method = config.javaClass.getDeclaredMethod("getClientUid")
                            val uid = method.invoke(config) as Int
                            if (uid > 0) {
                                val pkgs = packageManager.getPackagesForUid(uid)
                                pkgs?.firstOrNull()?.let { pkg ->
                                    if (pkg == packageName) null  // Skip self
                                    else try {
                                        val appInfo = packageManager.getApplicationInfo(pkg, 0)
                                        packageManager.getApplicationLabel(appInfo).toString()
                                    } catch (_: PackageManager.NameNotFoundException) { pkg }
                                }
                            } else null
                        } catch (_: Throwable) { null }
                    }.distinct()

                    val appStr = apps.joinToString(", ")
                    if (appStr != sourceAppName) {
                        sourceAppName = appStr
                        withContext(Dispatchers.Main) {
                            val status = if (appStr.isNotBlank()) "媒体转译中 · $appStr" else "媒体转译中"
                            updateStatus(status)
                            updateNotification(status)
                            asrCallback?.onStateChanged(true, status, appStr)
                        }
                    }
                } catch (_: Throwable) {}
                delay(3000)
            }
        }
    }

    // ===================== Status =====================

    private fun updateStatus(status: String) { captureStatus = status }

    private fun reportError(msg: String) {
        lastError = msg
        captureStatus = "错误: $msg"
        Log.e(TAG, msg)
        asrCallback?.onError(msg)
        asrCallback?.onStateChanged(false, "错误: $msg", "")
        updateNotification("错误: $msg")
    }

    // ===================== Notification =====================

    private fun createNotificationChannel() {
        val channel = NotificationChannel(CHANNEL_ID, "媒体音频捕获", NotificationManager.IMPORTANCE_LOW).apply {
            description = "正在捕获系统音频进行翻译"
        }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun buildNotification(text: String): Notification {
        val pi = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val stopPi = PendingIntent.getService(this, 1,
            Intent(this, MediaCaptureService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("媒体转译")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(pi)
            .addAction(Notification.Action.Builder(null, "停止", stopPi).build())
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(text: String) {
        try {
            getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, buildNotification(text))
        } catch (_: Throwable) {}
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

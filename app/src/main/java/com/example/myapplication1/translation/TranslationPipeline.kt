package com.example.myapplication1.translation

import android.util.Log
import kotlinx.coroutines.*
import java.util.LinkedHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Minimal translation pipeline: translate each sentence ASAP, TTS ASAP.
 * No ordering, no queuing, no semaphores — pure speed.
 *
 * Features:
 * - LRU translation cache: repeated phrases hit cache instantly (0ms latency).
 * - Paragraph refinement is fire-and-forget background work that only updates display.
 */
class TranslationPipeline(private val scope: CoroutineScope) {

    companion object {
        private const val TAG = "TransPipeline"

        /** Maximum number of entries in the translation LRU cache. */
        private const val CACHE_MAX_SIZE = 200

        /** Time-to-live for closing paragraphs — forced cleanup if stuck. */
        private const val PARAGRAPH_TTL_MS = 120_000L

        /** Minimum interval between TTL cleanup scans. */
        private const val CLEANUP_THROTTLE_MS = 30_000L

        /** Normalize English text for cache key: lowercase, collapse whitespace. */
        @JvmStatic
        fun normalizeCacheKey(text: String): String =
            text.trim().lowercase().replace(Regex("""\s+"""), " ")
    }

    interface Callback {
        fun onTranslationStarted(seqId: Int, en: String)
        fun onTranslationResult(seqId: Int, en: String, zh: String)
        fun onTranslationError(seqId: Int, en: String, error: String)
        fun onTtsReady(zh: String)
        fun onLatencyMeasured(translationMs: Long)
        fun onCacheHit(seqId: Int)
        fun onParagraphRefined(paragraphId: Int, refinedZh: String)
        /** SWR quality upgrade: a better translation arrived after the fast result. */
        fun onTranslationUpgraded(seqId: Int, en: String, zh: String, meta: TranslationMeta) {}
    }

    private val seqCounter = AtomicInteger(0)
    private val pendingCount = AtomicInteger(0)

    // LRU cache: normalized English → Chinese translation
    private val translationCache: LinkedHashMap<String, String> =
        object : LinkedHashMap<String, String>(16, 0.75f, true) {
            override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, String>?) =
                size > CACHE_MAX_SIZE
        }

    // Cache statistics
    private val cacheHits = AtomicLong(0)
    private val cacheMisses = AtomicLong(0)

    // Paragraph: just stores (en,zh) pairs for later refinement — no blocking
    private val paragraphData = mutableMapOf<Int, MutableList<Pair<String, String>>>()

    // Paragraph state machine: track in-flight translations per paragraph
    private val paragraphPendingCount = mutableMapOf<Int, Int>()
    private val paragraphClosing = mutableSetOf<Int>()
    private val paragraphClosedAt = mutableMapOf<Int, Long>()
    private val paragraphRefining = mutableSetOf<Int>()
    @Volatile private var lastCleanupMs = 0L

    @Volatile private var callback: Callback? = null
    private var engine: TranslationEngine? = null
    private var qualityEngine: TranslationEngine? = null
    private var refiner: TranslationRefiner? = null

    // Context for background/glossary/latency mode
    @Volatile var translationContext: TranslationContext = TranslationContext()

    @Volatile var enableTranslation: Boolean = true

    /** Timeout for quality path in BALANCED mode (ms). */
    private val qualityTimeoutMs = 8_000L

    val pendingTranslations: Int get() = pendingCount.get()
    val cacheHitCount: Long get() = cacheHits.get()
    val cacheMissCount: Long get() = cacheMisses.get()

    fun setCallback(cb: Callback?) { callback = cb }
    fun setEngine(eng: TranslationEngine?) { engine = eng }
    fun setQualityEngine(eng: TranslationEngine?) { qualityEngine = eng }
    fun setRefiner(ref: TranslationRefiner?) { refiner = ref }

    /** Allocate a seqId so caller can update UI state before translation starts. */
    fun allocateSeqId(): Int = seqCounter.getAndIncrement()

    /**
     * Evict the translation cache. Call when engine changes or context shifts significantly.
     */
    fun clearCache() {
        synchronized(translationCache) { translationCache.clear() }
        cacheHits.set(0)
        cacheMisses.set(0)
        Log.d(TAG, "Translation cache cleared")
    }

    /**
     * Start translating a sentence with a pre-allocated seqId.
     * Caller MUST have already updated UI state with this seqId.
     *
     * Cache lookup is performed on the calling thread (Main) for zero latency on hit.
     * On a cache miss the translation is dispatched to IO and results are delivered async.
     */
    fun submitSentence(seqId: Int, paragraphId: Int, en: String) {
        if (!enableTranslation) {
            scope.launch(Dispatchers.Main) {
                callback?.onTranslationResult(seqId, en, "")
                callback?.onLatencyMeasured(0L)
            }
            return
        }

        // ---- Fast path: cache lookup on calling thread ----
        val key = normalizeCacheKey(en)
        val cached: String? = synchronized(translationCache) { translationCache[key] }
        if (cached != null) {
            cacheHits.incrementAndGet()
            Log.d(TAG, "Cache hit [$key] → $cached")
            // Deliver on Main (we are already on Main, but stay consistent with async path)
            scope.launch(Dispatchers.Main) {
                callback?.onTranslationResult(seqId, en, cached)
                callback?.onLatencyMeasured(0L)
                callback?.onCacheHit(seqId)
                callback?.onTtsReady(cached)
            }
            synchronized(paragraphData) {
                paragraphData.getOrPut(paragraphId) { mutableListOf() }.add(en to cached)
            }
            return
        }

        // ---- Slow path: call translation engine ----
        cacheMisses.incrementAndGet()
        pendingCount.incrementAndGet()
        incParagraphPending(paragraphId)

        // Build effective context with domain-resolved glossary terms
        val ctx = resolveContext(en)

        scope.launch(Dispatchers.IO) {
            var fastZh = ""
            try {
                val eng = engine ?: throw Exception("No translation engine")
                val t0 = System.currentTimeMillis()
                val zh = eng.translate(en, ctx)
                val ms = System.currentTimeMillis() - t0
                fastZh = zh

                // Populate cache for future identical sentences
                if (zh.isNotBlank()) {
                    synchronized(translationCache) { translationCache[key] = zh }
                }

                // Store for later paragraph refinement
                synchronized(paragraphData) {
                    paragraphData.getOrPut(paragraphId) { mutableListOf() }.add(en to zh)
                }

                // Fire callbacks immediately — no ordering wait
                Log.d(TAG, "翻译完成 seq=$seqId ${ms}ms: ${en.take(30)} → ${zh.take(30)}")
                withContext(Dispatchers.Main) {
                    callback?.onTranslationResult(seqId, en, zh)
                    callback?.onLatencyMeasured(ms)
                    callback?.onTtsReady(zh)
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Translation failed: ${e.message}")
                withContext(Dispatchers.Main) {
                    callback?.onTranslationError(seqId, en, e.message ?: "error")
                }
                synchronized(paragraphData) {
                    paragraphData.getOrPut(paragraphId) { mutableListOf() }.add(en to "")
                }
            } finally {
                pendingCount.decrementAndGet()
                decParagraphPending(paragraphId)
                maybeFinalizeParagraph(paragraphId)
            }

            // ---- SWR quality path (fire-and-forget) ----
            maybeFireQualityPath(seqId, paragraphId, en, fastZh, ctx)
        }

        // Throttled TTL cleanup
        val now = System.currentTimeMillis()
        if (now - lastCleanupMs > CLEANUP_THROTTLE_MS) {
            lastCleanupMs = now
            cleanupExpiredParagraphs(now)
        }
    }

    /**
     * Mark a paragraph as closing and trigger refinement.
     * Does NOT immediately clean up paragraphData — cleanup is deferred until
     * all in-flight translations complete and refinement finishes.
     * Duplicate calls for the same paragraphId are safe (no-op).
     */
    fun closeParagraph(paragraphId: Int) {
        synchronized(paragraphData) {
            if (paragraphId in paragraphClosing) return          // idempotent
            paragraphClosing.add(paragraphId)
            paragraphClosedAt[paragraphId] = System.currentTimeMillis()
        }

        val ref = refiner
        val pairs = synchronized(paragraphData) {
            paragraphData[paragraphId]?.filter { it.second.isNotBlank() }?.toList()
        }

        // No refiner or no data → skip refinement, attempt finalize
        if (ref == null || pairs.isNullOrEmpty()) {
            scope.launch(Dispatchers.Main) {
                callback?.onParagraphRefined(paragraphId, "")
            }
            maybeFinalizeParagraph(paragraphId)
            return
        }

        synchronized(paragraphData) { paragraphRefining.add(paragraphId) }

        scope.launch(Dispatchers.IO) {
            val refined = try {
                ref.refineParagraph(pairs)
            } catch (e: Throwable) {
                Log.w(TAG, "Refinement failed: ${e.message}")
                ""
            }
            synchronized(paragraphData) { paragraphRefining.remove(paragraphId) }
            withContext(Dispatchers.Main) {
                callback?.onParagraphRefined(paragraphId, refined)
            }
            maybeFinalizeParagraph(paragraphId)
        }
    }

    // ==================== Context & Quality path ====================

    /**
     * Build the effective [TranslationContext] for a sentence:
     * resolve domain from hint or auto-detection, attach glossary terms.
     */
    private fun resolveContext(en: String): TranslationContext {
        val base = translationContext
        val domain = GlossaryManager.resolveDomain(base.domainHint, en)
        val terms = GlossaryManager.getTerms(domain)
        return base.copy(glossaryTerms = if (terms.isNotEmpty()) terms else base.glossaryTerms)
    }

    /**
     * SWR quality path: if a quality engine is configured and latency mode allows it,
     * fire a background quality translation. The result is delivered via
     * [Callback.onTranslationUpgraded] — it never triggers TTS (avoids double speak).
     *
     * Quality path is fire-and-forget: failures are silently logged.
     * It does NOT count toward [pendingCount] or [paragraphPendingCount] to avoid
     * blocking paragraph finalization.
     */
    private suspend fun maybeFireQualityPath(
        seqId: Int, paragraphId: Int, en: String, fastZh: String, ctx: TranslationContext
    ) {
        val qEng = qualityEngine
        if (qEng == null) {
            Log.d(TAG, "QualityPath seq=$seqId SKIP reason=no_quality_engine")
            return
        }
        if (ctx.latencyMode == LatencyMode.REALTIME) {
            Log.d(TAG, "QualityPath seq=$seqId SKIP reason=realtime_mode")
            return
        }
        if (fastZh.isBlank()) {
            Log.d(TAG, "QualityPath seq=$seqId SKIP reason=fast_path_empty")
            return
        }

        Log.d(TAG, "QualityPath seq=$seqId TRIGGERED mode=${ctx.latencyMode}")
        val t0 = System.currentTimeMillis()
        try {
            val qualityCtx = ctx.copy(latencyMode = LatencyMode.QUALITY)
            val qualZh = if (ctx.latencyMode == LatencyMode.BALANCED) {
                val result = withTimeoutOrNull(qualityTimeoutMs) { qEng.translate(en, qualityCtx) }
                if (result == null) Log.w(TAG, "QualityPath seq=$seqId TIMEOUT after ${qualityTimeoutMs}ms")
                result
            } else {
                qEng.translate(en, qualityCtx)
            }

            val ms = System.currentTimeMillis() - t0
            if (qualZh != null && qualZh.isNotBlank() && qualZh != fastZh) {
                val key = normalizeCacheKey(en)
                synchronized(translationCache) { translationCache[key] = qualZh }

                synchronized(paragraphData) {
                    paragraphData[paragraphId]?.let { pairs ->
                        val idx = pairs.indexOfLast { it.first == en }
                        if (idx >= 0) pairs[idx] = en to qualZh
                    }
                }

                val domain = GlossaryManager.resolveDomain(ctx.domainHint, en)
                val meta = TranslationMeta(
                    route = "quality",
                    selectedDomain = domain,
                    selectedGlossary = domain,
                    usedBackground = ctx.background.isNotBlank()
                )
                Log.d(TAG, "QualityPath seq=$seqId SUCCESS ${ms}ms: ${fastZh.take(20)} → ${qualZh.take(20)}")
                withContext(Dispatchers.Main) {
                    callback?.onTranslationUpgraded(seqId, en, qualZh, meta)
                }
            } else {
                val reason = when {
                    qualZh == null -> "timeout_or_null"
                    qualZh.isBlank() -> "empty_result"
                    else -> "same_as_fast"
                }
                Log.d(TAG, "QualityPath seq=$seqId NOOP ${ms}ms reason=$reason")
            }
        } catch (e: Throwable) {
            val ms = System.currentTimeMillis() - t0
            Log.w(TAG, "QualityPath seq=$seqId FAILED ${ms}ms: ${e.message}")
        }
    }

    // ==================== Paragraph state helpers ====================

    private fun incParagraphPending(paragraphId: Int) {
        synchronized(paragraphData) {
            paragraphPendingCount[paragraphId] = (paragraphPendingCount[paragraphId] ?: 0) + 1
        }
    }

    private fun decParagraphPending(paragraphId: Int) {
        synchronized(paragraphData) {
            val old = paragraphPendingCount[paragraphId] ?: 0
            paragraphPendingCount[paragraphId] = maxOf(0, old - 1)
        }
    }

    /**
     * Finalize a paragraph if all conditions are met:
     * - paragraph is marked as closing
     * - no in-flight translations (pending == 0)
     * - not currently being refined
     */
    private fun maybeFinalizeParagraph(paragraphId: Int) {
        synchronized(paragraphData) {
            if (paragraphId !in paragraphClosing) return
            if ((paragraphPendingCount[paragraphId] ?: 0) > 0) return
            if (paragraphId in paragraphRefining) return

            paragraphData.remove(paragraphId)
            paragraphPendingCount.remove(paragraphId)
            paragraphClosing.remove(paragraphId)
            paragraphClosedAt.remove(paragraphId)
            Log.d(TAG, "Paragraph $paragraphId finalized and cleaned up")
        }
    }

    /**
     * Force-clean paragraphs that have been in closing state longer than [PARAGRAPH_TTL_MS].
     * Acts as a safety net against stuck state from unexpected failures.
     */
    fun cleanupExpiredParagraphs(nowMs: Long = System.currentTimeMillis()) {
        synchronized(paragraphData) {
            val expired = paragraphClosedAt.filter { nowMs - it.value > PARAGRAPH_TTL_MS }.keys.toList()
            for (id in expired) {
                Log.w(TAG, "TTL forced cleanup for paragraph $id")
                paragraphData.remove(id)
                paragraphPendingCount.remove(id)
                paragraphClosing.remove(id)
                paragraphClosedAt.remove(id)
                paragraphRefining.remove(id)
            }
        }
    }

    /**
     * Reset state for a new session (e.g. "clear all" button).
     * Keeps the LRU translation cache alive: common phrases from prior sessions
     * are still valid translations regardless of session context.
     */
    fun reset() {
        // NOTE: Do NOT null callback — it must survive session resets.
        // The callback is the bridge to MainActivity's UI; nulling it
        // causes all subsequent translations to be silently discarded.
        seqCounter.set(0)
        pendingCount.set(0)
        synchronized(paragraphData) {
            paragraphData.clear()
            paragraphPendingCount.clear()
            paragraphClosing.clear()
            paragraphClosedAt.clear()
            paragraphRefining.clear()
        }
    }

    /**
     * Release all resources.  Unlike [reset], this also clears the cache because
     * the Activity is being destroyed and the pipeline will be reconstructed.
     * Clearing avoids returning stale results to a future pipeline instance
     * that may use a different engine.
     */
    fun close() {
        callback = null; engine = null; qualityEngine = null; refiner = null
        seqCounter.set(0); pendingCount.set(0)
        synchronized(paragraphData) {
            paragraphData.clear()
            paragraphPendingCount.clear()
            paragraphClosing.clear()
            paragraphClosedAt.clear()
            paragraphRefining.clear()
        }
        synchronized(translationCache) { translationCache.clear() }
    }
}

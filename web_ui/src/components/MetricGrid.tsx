import type { InferenceMetrics } from '../types'
import { eyeConfidence, formatNumber } from '../utils/format'

type MetricGridProps = {
  metrics: InferenceMetrics
  blinkRate: number
  lastBlinkAgo: number | null
  latencyMs: number | null
}

export function MetricGrid({ metrics, blinkRate, lastBlinkAgo, latencyMs }: MetricGridProps) {
  return (
    <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
      <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-neutral-500">Eye state</p>
        <p className="mt-2 text-3xl font-semibold">{metrics.eyeState === 'CLOSED' ? 'Closed' : 'Open'}</p>
        <p className="mt-1 text-sm text-neutral-500">
          Confidence {formatNumber(eyeConfidence(metrics.confidence, metrics.eyeState) * 100, 0)}%
        </p>
      </div>

      <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-neutral-500">Blinks</p>
        <p className="mt-2 text-3xl font-semibold">{metrics.totalBlinks}</p>
        <p className="mt-1 text-sm text-neutral-500">
          {blinkRate > 0 ? `${formatNumber(blinkRate)} / min` : 'Calculating rate'}
        </p>
      </div>

      <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-neutral-500">Frame rate</p>
        <p className="mt-2 text-3xl font-semibold">{formatNumber(metrics.fps)} fps</p>
        <p className="mt-1 text-sm text-neutral-500">Buffer {metrics.bufferFull ? 'ready' : 'warming up'}</p>
      </div>

      <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-neutral-500">Face lock</p>
        <p className="mt-2 text-3xl font-semibold">{metrics.faceDetected ? 'Tracked' : 'Searching'}</p>
        <p className="mt-1 text-sm text-neutral-500">
          {metrics.bufferFull ? 'Stable stream' : 'Need more frames'}
        </p>
      </div>

      <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-neutral-500">Last blink</p>
        <p className="mt-2 text-3xl font-semibold">
          {lastBlinkAgo ? `${formatNumber(lastBlinkAgo)}s ago` : '—'}
        </p>
        <p className="mt-1 text-sm text-neutral-500">Recent events</p>
      </div>

      <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-neutral-500">Latency</p>
        <p className="mt-2 text-3xl font-semibold">
          {latencyMs !== null ? `${formatNumber(latencyMs, 0)} ms` : '—'}
        </p>
        <p className="mt-1 text-sm text-neutral-500">Round trip to inference API</p>
      </div>
    </div>
  )
}

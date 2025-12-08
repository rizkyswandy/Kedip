import type { InferenceMetrics } from '../types'
import { eyeConfidence, formatNumber } from '../utils/format'

type InferencePanelProps = {
  metrics: InferenceMetrics
}

export function InferencePanel({ metrics }: InferencePanelProps) {
  return (
    <div className="rounded-3xl border border-neutral-200 bg-white p-5 shadow-[0_20px_80px_-60px_rgba(0,0,0,0.35)]">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-neutral-500">Inference stream</p>
          <p className="text-sm text-neutral-500">Live metrics from your local model</p>
        </div>
        <div className="rounded-full bg-neutral-900 px-3 py-1 text-xs font-semibold text-white">
          {metrics.bufferFull ? 'Ready' : 'Syncing'}
        </div>
      </div>

      <div className="mt-4 space-y-3 text-sm text-neutral-600">
        <div className="flex items-center justify-between rounded-2xl border border-neutral-100 px-3 py-2">
          <span>Status</span>
          <span className="font-semibold text-neutral-900">{metrics.faceDetected ? 'Face detected' : 'Waiting'}</span>
        </div>
        <div className="flex items-center justify-between rounded-2xl border border-neutral-100 px-3 py-2">
          <span>Buffer</span>
          <span className="font-semibold text-neutral-900">{metrics.bufferFull ? 'Full' : 'Filling'}</span>
        </div>
        <div className="flex items-center justify-between rounded-2xl border border-neutral-100 px-3 py-2">
          <span>Confidence</span>
          <span className="font-semibold text-neutral-900">
            {formatNumber(eyeConfidence(metrics.confidence, metrics.eyeState) * 100, 0)}%
          </span>
        </div>
        <div className="flex items-center justify-between rounded-2xl border border-neutral-100 px-3 py-2">
          <span>FPS</span>
          <span className="font-semibold text-neutral-900">{formatNumber(metrics.fps)} fps</span>
        </div>
      </div>
    </div>
  )
}

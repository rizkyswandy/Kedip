import type { RefObject } from 'react'

type CameraPanelProps = {
  status: string
  isCameraOn: boolean
  videoRef: RefObject<HTMLVideoElement | null>
  onStart: () => void
  onStop: () => void
  onReset: () => void
}

export function CameraPanel({ status, isCameraOn, videoRef, onStart, onStop, onReset }: CameraPanelProps) {
  return (
    <div className="overflow-hidden rounded-3xl border border-neutral-200 bg-white shadow-[0_20px_80px_-60px_rgba(0,0,0,0.35)]">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-100 px-5 py-4">
        <div className="space-y-1">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-neutral-500">Live camera</p>
          <p className="text-sm text-neutral-500">{status}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={onStart}
            className="rounded-full border border-neutral-900 px-4 py-2 text-sm font-semibold text-neutral-900 transition hover:-translate-y-0.5 hover:shadow-md focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-900 disabled:cursor-not-allowed disabled:border-neutral-200 disabled:text-neutral-400"
            disabled={isCameraOn}
          >
            Start camera
          </button>
          <button
            onClick={onStop}
            className="rounded-full bg-neutral-900 px-4 py-2 text-sm font-semibold text-white transition hover:-translate-y-0.5 hover:shadow-md focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-900 disabled:cursor-not-allowed disabled:bg-neutral-200"
            disabled={!isCameraOn}
          >
            Stop
          </button>
          <button
            onClick={onReset}
            className="rounded-full border border-neutral-200 px-4 py-2 text-sm font-semibold text-neutral-700 transition hover:-translate-y-0.5 hover:shadow-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-900"
          >
            Reset stats
          </button>
        </div>
      </div>

      <div className="relative bg-neutral-50">
        <div className="aspect-[16/9] w-full">
          <video ref={videoRef} className="h-full w-full object-cover" playsInline muted autoPlay />
        </div>

        {!isCameraOn && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/80 backdrop-blur">
            <div className="rounded-2xl border border-neutral-200 px-6 py-5 text-center shadow-sm">
              <p className="text-sm font-medium text-neutral-800">Camera is idle</p>
              <p className="text-sm text-neutral-500">Start the camera to stream frames to the local model.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

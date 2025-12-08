import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { CameraPanel } from './components/CameraPanel'
import { ErrorBanner } from './components/ErrorBanner'
import { Header } from './components/Header'
import { InferencePanel } from './components/InferencePanel'
import { MetricGrid } from './components/MetricGrid'
import type { InferenceMetrics, RawInferenceResponse } from './types'

const API_BASE = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/$/, '')
const CAPTURE_INTERVAL_MS = 150
const DEFAULT_VIDEO_WIDTH = 480

const initialMetrics: InferenceMetrics = {
  hasBlink: false,
  confidence: 0,
  eyeState: 'OPEN',
  totalBlinks: 0,
  fps: 0,
  faceDetected: false,
  bufferFull: false,
}

function App() {
  const [metrics, setMetrics] = useState<InferenceMetrics>(initialMetrics)
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [status, setStatus] = useState('Ready to start capture')
  const [error, setError] = useState<string | null>(null)
  const [blinkHistory, setBlinkHistory] = useState<number[]>([])
  const [apiHealthy, setApiHealthy] = useState<boolean | null>(null)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)

  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const captureIntervalRef = useRef<number | null>(null)
  const sendingRef = useRef(false)
  const startTimeRef = useRef<number | null>(null)

  const sessionId = useMemo(
    () => (crypto.randomUUID ? crypto.randomUUID() : `session-${Date.now().toString(36)}`),
    []
  )

  const blinkRate = useMemo(() => {
    if (!startTimeRef.current) return 0
    const elapsedMinutes = (Date.now() - startTimeRef.current) / 60000
    if (elapsedMinutes <= 0) return 0
    return metrics.totalBlinks / elapsedMinutes
  }, [metrics.totalBlinks])

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`)
      setApiHealthy(res.ok)
    } catch {
      setApiHealthy(false)
    }
  }, [])

  const stopCamera = useCallback(() => {
    if (captureIntervalRef.current) {
      window.clearInterval(captureIntervalRef.current)
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }

    sendingRef.current = false
    startTimeRef.current = null
    setLatencyMs(null)
    setIsCameraOn(false)
    setStatus('Camera stopped')
  }, [])

  const resetSession = useCallback(async () => {
    setMetrics(initialMetrics)
    setBlinkHistory([])
    startTimeRef.current = isCameraOn ? Date.now() : null
    setLatencyMs(null)
    setStatus('Session reset')

    try {
      await fetch(`${API_BASE}/api/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      })
    } catch {
      // Non-blocking: reset locally even if API reset fails
    }
  }, [isCameraOn, sessionId])

  const startCamera = useCallback(async () => {
    setError(null)
    setStatus('Requesting camera access...')

    await checkHealth()

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      streamRef.current = stream
      startTimeRef.current = Date.now()
      setMetrics(initialMetrics)
      setBlinkHistory([])
      setLatencyMs(null)
      setIsCameraOn(true)
      setStatus('Buffering frames...')
    } catch (err) {
      console.error(err)
      setError('Could not access your camera. Check permissions and try again.')
      setStatus('Camera unavailable')
      stopCamera()
    }
  }, [checkHealth, stopCamera])

  const normalizeResponse = (payload: RawInferenceResponse): InferenceMetrics => {
    const hasBlink = payload.has_blink ?? payload.hasBlink ?? false
    const totalBlinks = payload.total_blinks ?? payload.totalBlinks ?? 0
    const rawConfidence = payload.confidence ?? 0
    const confidence = Math.min(Math.max(rawConfidence, 0), 1)
    const eyeValue = payload.eye_state ?? payload.eyeState ?? 0
    const eyeState = typeof eyeValue === 'string'
      ? (eyeValue.toUpperCase() === 'CLOSED' ? 'CLOSED' : 'OPEN')
      : eyeValue > 0.5
        ? 'CLOSED'
        : 'OPEN'

    return {
      hasBlink,
      confidence,
      eyeState,
      totalBlinks,
      fps: payload.fps ?? 0,
      faceDetected: payload.face_detected ?? payload.faceDetected ?? false,
      bufferFull: payload.buffer_full ?? payload.bufferFull ?? false,
    }
  }

  const sendFrame = useCallback(async () => {
    if (!isCameraOn || !videoRef.current) return
    if (sendingRef.current) return

    const video = videoRef.current
    if (video.readyState < 2) return

    const canvas = canvasRef.current ?? document.createElement('canvas')
    const context = canvas.getContext('2d')
    if (!context) return

    canvasRef.current = canvas

    const ratio =
      video.videoWidth && video.videoHeight
        ? video.videoWidth / video.videoHeight
        : 16 / 9
    const targetWidth = DEFAULT_VIDEO_WIDTH
    const targetHeight = targetWidth / ratio

    canvas.width = targetWidth
    canvas.height = targetHeight
    context.drawImage(video, 0, 0, targetWidth, targetHeight)

    // Compress the frame before sending for inference
    const imageData = canvas.toDataURL('image/jpeg', 0.5)

    sendingRef.current = true

    try {
      const started = performance.now()
      const res = await fetch(`${API_BASE}/api/infer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData, session_id: sessionId }),
      })

      if (!res.ok) {
        throw new Error('Inference API is unavailable.')
      }

      const payload: RawInferenceResponse = await res.json()
      const normalized = normalizeResponse(payload)
      setLatencyMs(performance.now() - started)

      const eventTimestampMs =
        payload.timestamp !== undefined ? payload.timestamp * 1000 : Date.now()

      setApiHealthy(true)
      setMetrics((prev) => {
        const updated = { ...prev, ...normalized }
        const blinkIncremented = normalized.totalBlinks > prev.totalBlinks
        const blinkDetected = normalized.hasBlink || blinkIncremented

        if (blinkDetected) {
          setBlinkHistory((prevHistory) => [...prevHistory.slice(-4), eventTimestampMs])
        }

        return updated
      })

      setStatus(normalized.bufferFull ? 'Analyzing in real time' : 'Warming up (collecting frames)')
      setError(null)
    } catch (err) {
      console.error(err)
      setApiHealthy(false)
      setLatencyMs(null)
      setError((err as Error).message ?? 'Could not reach inference API')
    } finally {
      sendingRef.current = false
    }
  }, [isCameraOn, sessionId])

  useEffect(() => {
    if (!isCameraOn) return

    captureIntervalRef.current = window.setInterval(() => {
      void sendFrame()
    }, CAPTURE_INTERVAL_MS)

    return () => {
      if (captureIntervalRef.current) {
        window.clearInterval(captureIntervalRef.current)
      }
    }
  }, [isCameraOn, sendFrame])

  useEffect(() => {
    return () => stopCamera()
  }, [stopCamera])

  const lastBlink = blinkHistory.at(-1)
  const lastBlinkAgo = lastBlink ? Math.max(0, (Date.now() - lastBlink) / 1000) : null

  return (
    <div className="min-h-screen bg-white text-neutral-900">
      <div className="mx-auto flex max-w-6xl flex-col gap-8 px-4 py-10 md:px-8">
        <Header apiHealthy={apiHealthy} isCameraOn={isCameraOn} />

        <main className="grid gap-6 lg:grid-cols-[1.7fr,1fr]">
          <section className="space-y-4">
            <CameraPanel
              status={status}
              isCameraOn={isCameraOn}
              videoRef={videoRef}
              onStart={startCamera}
              onStop={stopCamera}
              onReset={resetSession}
            />

            {error && <ErrorBanner message={error} />}

            <MetricGrid
              metrics={metrics}
              blinkRate={blinkRate}
              lastBlinkAgo={lastBlinkAgo}
              latencyMs={latencyMs}
            />
          </section>

          <aside className="space-y-4">
            <InferencePanel metrics={metrics} />
          </aside>
        </main>
      </div>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}

export default App

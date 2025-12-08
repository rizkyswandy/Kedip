export type RawInferenceResponse = {
  has_blink?: boolean
  hasBlink?: boolean
  confidence?: number
  eye_state?: number | string
  eyeState?: number | string
  total_blinks?: number
  totalBlinks?: number
  fps?: number
  face_detected?: boolean
  faceDetected?: boolean
  buffer_full?: boolean
  bufferFull?: boolean
  timestamp?: number
}

export type InferenceMetrics = {
  hasBlink: boolean
  confidence: number
  eyeState: 'OPEN' | 'CLOSED'
  totalBlinks: number
  fps: number
  faceDetected: boolean
  bufferFull: boolean
}

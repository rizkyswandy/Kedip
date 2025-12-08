export const formatNumber = (value: number, digits = 1) =>
  Number.isFinite(value) ? value.toFixed(digits) : '0'

export const eyeConfidence = (confidence: number, eyeState: 'OPEN' | 'CLOSED') => {
  const value = eyeState === 'CLOSED' ? confidence : 1 - confidence
  return Math.min(Math.max(value, 0), 1)
}

import type { ModelInfo, ModelKey } from '../../app/types'

export const MODELS: Record<ModelKey, ModelInfo> = {
  tiny: {
    key: 'tiny',
    label: 'Whisper Tiny (39M)',
    repo: 'onnx-community/whisper-tiny.en_timestamped',
    params: '39M',
    realtimeCapable: true,
  },
  base: {
    key: 'base',
    label: 'Whisper Base (74M)',
    repo: 'onnx-community/whisper-base.en_timestamped',
    params: '74M',
    realtimeCapable: true,
  },
  small: {
    key: 'small',
    label: 'Whisper Small (244M)',
    repo: 'onnx-community/whisper-small.en_timestamped',
    params: '244M',
    realtimeCapable: true,
  },
}

export const MODEL_DTYPE = {
  encoder_model: 'fp32',
  decoder_model_merged: 'q4',
} as const

export const MODEL_KEYS = Object.keys(MODELS) as ModelKey[]

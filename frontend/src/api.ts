import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export default api

export interface DatasetInfo {
  total_samples: number
  train_samples: number
  val_samples: number
  test_samples: number
  n_classes: number
  class_names: string[]
  class_distribution: Record<string, number>
  n_months: number
  n_features: number
  feature_names: string[]
  s1_features: string[]
  s2_features: string[]
}

export interface Sample {
  sample_index: number
  id: string
  label: string
  label_index: number
}

export interface MonthlyObservation {
  month: number
  s1: { VV: number; VH: number }
  s2: Record<string, number>
}

export interface SampleDetail extends Sample {
  monthly_observations: MonthlyObservation[]
}

export interface PredictionResult {
  sample_index: number
  model: string
  predicted_class: string
  predicted_index: number
  true_class: string
  true_index: number
  correct: boolean
  probabilities: Record<string, number>
  confidence: number
  missing_months_s1: number[]
  missing_months_s2: number[]
}

export interface ModelStatus {
  available: boolean
  epoch?: number
  val_acc?: number
}

export interface HealthStatus {
  status: string
  device: string
  checkpoints: Record<string, boolean>
  experiments_ready: boolean
  data_cache_exists: boolean
}

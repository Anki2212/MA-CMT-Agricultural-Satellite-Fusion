import { useEffect, useState, useCallback } from 'react'
import api from '../api'
import type { Sample, SampleDetail, PredictionResult } from '../api'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts'

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
const MODEL_OPTIONS = [
  { value: 'macmt',  label: 'MA-CMT (Proposed)' },
  { value: 's1only', label: 'S1-Only' },
  { value: 's2only', label: 'S2-Only' },
  { value: 'concat', label: 'Concat Fusion' },
]

export default function CropPrediction() {
  const [samples, setSamples] = useState<Sample[]>([])
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null)
  const [detail, setDetail] = useState<SampleDetail | null>(null)
  const [model, setModel] = useState('macmt')
  const [missingS1, setMissingS1] = useState<Set<number>>(new Set())
  const [missingS2, setMissingS2] = useState<Set<number>>(new Set())
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [predLoading, setPredLoading] = useState(false)
  const [predError, setPredError] = useState<string | null>(null)
  const [classFilter, setClassFilter] = useState('')
  const [classes, setClasses] = useState<string[]>([])

  const loadSamples = useCallback((cls: string) => {
    const params = cls ? { split: 'test', class_filter: cls, limit: 100 } : { split: 'test', limit: 100 }
    api.get('/dataset/samples', { params }).then(r => setSamples(r.data.samples))
  }, [])

  useEffect(() => {
    api.get('/dataset/info').then(r => setClasses(r.data.class_names))
    loadSamples('')
  }, [loadSamples])

  const selectSample = async (idx: number) => {
    setSelectedIdx(idx)
    setResult(null)
    setMissingS1(new Set())
    setMissingS2(new Set())
    const r = await api.get(`/dataset/sample/${idx}`)
    setDetail(r.data)
  }

  const toggleMonth = (t: number, sensor: 's1' | 's2') => {
    const setter = sensor === 's1' ? setMissingS1 : setMissingS2
    setter(prev => {
      const next = new Set(prev)
      if (next.has(t)) {
        next.delete(t)
      } else {
        next.add(t)
      }
      return next
    })
    setResult(null)
  }

  const toggleSensor = (sensor: 's1' | 's2') => {
    const setter = sensor === 's1' ? setMissingS1 : setMissingS2
    setter(prev => {
      if (prev.size === 12) return new Set()
      return new Set(Array.from({ length: 12 }, (_, i) => i))
    })
    setResult(null)
  }

  const runPrediction = async () => {
    if (selectedIdx === null) return
    setPredLoading(true)
    setPredError(null)
    try {
      const r = await api.post('/predict', {
        sample_index: selectedIdx,
        model_name: model,
        missing_months_s1: Array.from(missingS1),
        missing_months_s2: Array.from(missingS2),
      })
      setResult(r.data)
    } catch (e: any) {
      setPredError(e.response?.data?.detail ?? e.message)
    } finally {
      setPredLoading(false)
    }
  }

  const probData = result
    ? Object.entries(result.probabilities)
        .map(([cls, prob]) => ({ cls, prob: parseFloat((prob * 100).toFixed(1)) }))
        .sort((a, b) => b.prob - a.prob)
    : []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Crop Prediction</h1>
        <p className="text-slate-500 mt-1 text-sm">
          Select a test sample, toggle missing months/sensors, then run inference.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sample selector */}
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
          <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-3">
            1. Select Sample
          </h2>
          <select
            className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm mb-3"
            value={classFilter}
            onChange={e => { setClassFilter(e.target.value); loadSamples(e.target.value) }}
          >
            <option value="">All classes</option>
            {classes.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
          <div className="max-h-72 overflow-y-auto space-y-1 pr-1">
            {samples.map(s => (
              <button
                key={s.sample_index}
                onClick={() => selectSample(s.sample_index)}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                  selectedIdx === s.sample_index
                    ? 'bg-green-700 text-white'
                    : 'hover:bg-slate-50 text-slate-700'
                }`}
              >
                <span className="font-medium">#{s.sample_index}</span>
                <span className={`ml-2 text-xs ${selectedIdx === s.sample_index ? 'text-green-200' : 'text-slate-400'}`}>
                  {s.label}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Month toggles */}
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
          <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-3">
            2. Toggle Missing Data
          </h2>
          {!detail ? (
            <p className="text-sm text-slate-400">Select a sample first.</p>
          ) : (
            <>
              <div className="flex gap-2 mb-3">
                <button
                  onClick={() => toggleSensor('s1')}
                  className="text-xs px-2 py-1 rounded border border-blue-300 text-blue-700 hover:bg-blue-50"
                >
                  {missingS1.size === 12 ? 'Restore S1' : 'Drop All S1'}
                </button>
                <button
                  onClick={() => toggleSensor('s2')}
                  className="text-xs px-2 py-1 rounded border border-emerald-300 text-emerald-700 hover:bg-emerald-50"
                >
                  {missingS2.size === 12 ? 'Restore S2' : 'Drop All S2'}
                </button>
              </div>
              <div className="grid grid-cols-6 gap-1">
                {MONTHS.map((m, t) => {
                  const s1Missing = missingS1.has(t)
                  const s2Missing = missingS2.has(t)
                  return (
                    <div key={t} className="text-center">
                      <div className="text-xs text-slate-400 mb-1">{m}</div>
                      <button
                        onClick={() => toggleMonth(t, 's1')}
                        title="S1"
                        className={`w-full h-5 rounded text-xs font-bold mb-0.5 border ${
                          s1Missing
                            ? 'bg-red-100 border-red-300 text-red-500'
                            : 'bg-blue-100 border-blue-300 text-blue-700'
                        }`}
                      >S1</button>
                      <button
                        onClick={() => toggleMonth(t, 's2')}
                        title="S2"
                        className={`w-full h-5 rounded text-xs font-bold border ${
                          s2Missing
                            ? 'bg-red-100 border-red-300 text-red-500'
                            : 'bg-emerald-100 border-emerald-300 text-emerald-700'
                        }`}
                      >S2</button>
                    </div>
                  )
                })}
              </div>
              <p className="text-xs text-slate-400 mt-2">
                S1 missing: {missingS1.size}/12 · S2 missing: {missingS2.size}/12
              </p>
            </>
          )}
        </div>

        {/* Model + run */}
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm flex flex-col gap-4">
          <div>
            <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-3">
              3. Model &amp; Inference
            </h2>
            <select
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm"
              value={model}
              onChange={e => { setModel(e.target.value); setResult(null) }}
            >
              {MODEL_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
          <button
            disabled={selectedIdx === null || predLoading}
            onClick={runPrediction}
            className="w-full py-2.5 rounded-lg bg-green-700 text-white text-sm font-semibold
                       hover:bg-green-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {predLoading ? 'Running…' : 'Run Prediction'}
          </button>

          {predError && (
            <div className="p-3 rounded-lg bg-amber-50 border border-amber-200 text-amber-700 text-xs">
              {predError}
            </div>
          )}

          {result && (
            <div className="space-y-2">
              <div className={`p-3 rounded-lg border ${
                result.correct
                  ? 'bg-green-50 border-green-200 text-green-800'
                  : 'bg-red-50 border-red-200 text-red-800'
              }`}>
                <p className="font-semibold text-sm">
                  {result.correct ? '✓ Correct' : '✗ Incorrect'}
                </p>
                <p className="text-xs mt-0.5">
                  Predicted: <strong>{result.predicted_class}</strong>
                </p>
                <p className="text-xs">
                  True: <strong>{result.true_class}</strong>
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Confidence: {(result.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Probability chart */}
      {result && (
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-4">
            Class Probabilities
          </h2>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={probData} layout="vertical" margin={{ left: 16, right: 24 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="cls" width={120} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => `${v}%`} />
              <Bar dataKey="prob" radius={[0, 4, 4, 0]}>
                {probData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={entry.cls === result.predicted_class ? '#166534' : '#86efac'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Monthly observations table */}
      {detail && (
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm overflow-x-auto">
          <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-3">
            Monthly Observations — Sample #{detail.sample_index} ({detail.label})
          </h2>
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="pb-2 pr-3 font-semibold text-slate-500">Month</th>
                <th className="pb-2 pr-3 font-semibold text-blue-600">S1 VV</th>
                <th className="pb-2 pr-3 font-semibold text-blue-600">S1 VH</th>
                {Object.keys(detail.monthly_observations[0].s2).map(k => (
                  <th key={k} className="pb-2 pr-3 font-semibold text-emerald-600">{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {detail.monthly_observations.map((obs) => {
                const t = obs.month - 1
                const s1m = missingS1.has(t)
                const s2m = missingS2.has(t)
                return (
                  <tr key={obs.month} className="border-b border-slate-50 hover:bg-slate-50">
                    <td className="py-1 pr-3 font-medium text-slate-600">{MONTHS[t]}</td>
                    <td className={`py-1 pr-3 ${s1m ? 'line-through text-red-400' : ''}`}>
                      {obs.s1.VV.toFixed(2)}
                    </td>
                    <td className={`py-1 pr-3 ${s1m ? 'line-through text-red-400' : ''}`}>
                      {obs.s1.VH.toFixed(2)}
                    </td>
                    {Object.values(obs.s2).map((v, i) => (
                      <td key={i} className={`py-1 pr-3 ${s2m ? 'line-through text-red-400' : ''}`}>
                        {v.toFixed(0)}
                      </td>
                    ))}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

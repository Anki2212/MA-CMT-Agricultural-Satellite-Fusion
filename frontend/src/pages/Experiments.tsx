import { useEffect, useState } from 'react'
import api from '../api'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'

const MODEL_COLORS: Record<string, string> = {
  macmt:  '#166534',
  s1only: '#1d4ed8',
  s2only: '#d97706',
  concat: '#7c3aed',
}

const MODEL_LABELS: Record<string, string> = {
  macmt:  'MA-CMT',
  s1only: 'S1-Only',
  s2only: 'S2-Only',
  concat: 'Concat',
}

const MISSING_RATES = [0, 10, 20, 30, 50]
const SCENARIOS = ['random', 's1_missing', 's2_missing', 'consecutive']
const SCENARIO_LABELS: Record<string, string> = {
  random: 'Random Missing',
  s1_missing: 'S1 Sensor Missing',
  s2_missing: 'S2 Sensor Missing',
  consecutive: 'Consecutive Months',
}

export default function Experiments() {
  const [results, setResults] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeScenario, setActiveScenario] = useState('random')
  const [cmModel, setCmModel] = useState('macmt')
  const [cmData, setCmData] = useState<any>(null)

  useEffect(() => {
    api.get('/experiments/summary')
      .then(r => setResults(r.data.summary))
      .catch(e => setError(e.response?.data?.detail ?? e.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!results) return
    api.get('/experiments/confusion_matrix', {
      params: { model: cmModel, scenario: 'complete', missing_rate: 0 },
    }).then(r => setCmData(r.data)).catch(() => setCmData(null))
  }, [results, cmModel])

  if (loading) return <Loading />
  if (error) return (
    <div className="p-4 rounded-lg bg-amber-50 border border-amber-200 text-amber-700 text-sm">
      <strong>Experiments not yet available.</strong>
      <p className="mt-1">{error}</p>
      <p className="mt-1 text-xs">Run all models first, then execute <code className="font-mono bg-amber-100 px-1">python ml/experiments/run_experiments.py</code></p>
    </div>
  )

  // Build chart data per scenario
  const getChartData = (scenario: string) => {
    return MISSING_RATES.map(rate => {
      const row: any = { rate: `${rate}%` }
      for (const model of ['macmt', 's1only', 's2only', 'concat']) {
        const entry = (results as any[]).find(
          r => r.scenario === (rate === 0 && scenario !== 'complete' ? 'complete' : scenario)
            && r.model === model
            && r.missing_rate === rate
        ) ?? (results as any[]).find(
          r => r.scenario === 'complete' && r.model === model && r.missing_rate === 0 && rate === 0
        )
        if (entry) {
          row[model + '_acc'] = parseFloat((entry.accuracy * 100).toFixed(1))
          row[model + '_f1']  = parseFloat((entry.f1_macro * 100).toFixed(1))
        }
      }
      return row
    })
  }

  const completeRows = (results as any[]).filter(r => r.scenario === 'complete')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Experiments</h1>
        <p className="text-slate-500 mt-1 text-sm">
          Model comparison under different missing-data scenarios and rates.
        </p>
      </div>

      {/* Complete performance table */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm overflow-x-auto">
        <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-4">
          Complete Observations (0% Missing)
        </h2>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs text-slate-500 uppercase">
              <th className="pb-2 pr-4">Model</th>
              <th className="pb-2 pr-4">Accuracy</th>
              <th className="pb-2 pr-4">Macro-F1</th>
              <th className="pb-2 pr-4">Precision</th>
              <th className="pb-2">Recall</th>
            </tr>
          </thead>
          <tbody>
            {completeRows.map((r: any) => (
              <tr key={r.model} className="border-b border-slate-50">
                <td className="py-2 pr-4 font-medium" style={{ color: MODEL_COLORS[r.model] }}>
                  {MODEL_LABELS[r.model] ?? r.model}
                </td>
                <td className="py-2 pr-4">{r.accuracy != null ? (r.accuracy * 100).toFixed(1) + '%' : '—'}</td>
                <td className="py-2 pr-4">{r.f1_macro != null ? (r.f1_macro * 100).toFixed(1) + '%' : '—'}</td>
                <td className="py-2 pr-4">{r.precision != null ? (r.precision * 100).toFixed(1) + '%' : '—'}</td>
                <td className="py-2">{r.recall != null ? (r.recall * 100).toFixed(1) + '%' : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Scenario selector + accuracy chart */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
        <div className="flex flex-wrap gap-2 mb-4">
          {SCENARIOS.map(s => (
            <button
              key={s}
              onClick={() => setActiveScenario(s)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
                activeScenario === s
                  ? 'bg-green-700 text-white border-green-700'
                  : 'border-slate-200 text-slate-600 hover:bg-slate-50'
              }`}
            >
              {SCENARIO_LABELS[s]}
            </button>
          ))}
        </div>
        <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-3">
          Accuracy vs Missing Rate — {SCENARIO_LABELS[activeScenario]}
        </h2>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={getChartData(activeScenario)} margin={{ right: 16 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="rate" tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v) => `${v}%`} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {Object.entries(MODEL_LABELS).map(([key, label]) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key + '_acc'}
                name={label}
                stroke={MODEL_COLORS[key]}
                strokeWidth={2}
                dot={{ r: 4 }}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Macro-F1 chart */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
        <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-3">
          Macro-F1 vs Missing Rate — {SCENARIO_LABELS[activeScenario]}
        </h2>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={getChartData(activeScenario)} margin={{ right: 16 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="rate" tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v) => `${v}%`} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {Object.entries(MODEL_LABELS).map(([key, label]) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key + '_f1'}
                name={label}
                stroke={MODEL_COLORS[key]}
                strokeWidth={2}
                strokeDasharray="5 3"
                dot={{ r: 3 }}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Confusion matrix */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
        <div className="flex flex-wrap gap-2 items-center mb-4">
          <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mr-2">
            Confusion Matrix (complete, 0% missing)
          </h2>
          <select
            value={cmModel}
            onChange={e => setCmModel(e.target.value)}
            className="border border-slate-200 rounded px-2 py-1 text-xs"
          >
            {Object.entries(MODEL_LABELS).map(([k, v]) => (
              <option key={k} value={k}>{v}</option>
            ))}
          </select>
        </div>
        {cmData?.confusion_matrix ? (
          <ConfusionMatrix cm={cmData.confusion_matrix} labels={cmData.class_names} />
        ) : (
          <p className="text-sm text-slate-400">No data.</p>
        )}
      </div>

      {/* Comprehensive Benchmark Figure */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
        <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-3">
          Overall Benchmark Figure (All Scenarios)
        </h2>
        <div className="flex justify-center">
          <img
            src="/api/experiments/plot"
            alt="Accuracy vs Missing Rate (All Scenarios)"
            className="rounded-lg max-h-[420px] w-auto border border-slate-200 shadow-xs"
          />
        </div>
      </div>
    </div>
  )
}

function ConfusionMatrix({ cm, labels }: { cm: number[][]; labels: string[] }) {
  const maxVal = Math.max(...cm.flat())
  return (
    <div className="overflow-x-auto">
      <table className="text-xs border-collapse">
        <thead>
          <tr>
            <th className="p-1 text-slate-400 text-right">True ↓ Pred →</th>
            {labels.map(l => (
              <th key={l} className="p-1 font-medium text-slate-600" style={{ width: 70 }}>
                <span className="block truncate max-w-[64px]" title={l}>{l.slice(0, 8)}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cm.map((row, i) => (
            <tr key={i}>
              <td className="p-1 font-medium text-slate-600 text-right pr-2 whitespace-nowrap">
                {labels[i].slice(0, 12)}
              </td>
              {row.map((val, j) => {
                const intensity = maxVal > 0 ? val / maxVal : 0
                const bg = i === j
                  ? `rgba(22,101,52,${0.15 + intensity * 0.75})`
                  : `rgba(239,68,68,${intensity * 0.5})`
                return (
                  <td
                    key={j}
                    className="border border-slate-100 text-center p-1 font-mono"
                    style={{ backgroundColor: bg, minWidth: 44 }}
                  >
                    {val}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Loading() {
  return (
    <div className="flex items-center justify-center h-64 text-slate-400">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-green-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p className="text-sm">Loading experiment results…</p>
      </div>
    </div>
  )
}

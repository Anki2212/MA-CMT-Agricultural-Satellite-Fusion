import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import CropPrediction from './pages/CropPrediction'
import Experiments from './pages/Experiments'
import About from './pages/About'

const NAV = [
  { to: '/', label: 'Dashboard' },
  { to: '/predict', label: 'Crop Prediction' },
  { to: '/experiments', label: 'Experiments' },
  { to: '/about', label: 'About' },
]

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col">
        {/* Header */}
        <header className="bg-green-900 text-white shadow-md">
          <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
            <div>
              <span className="text-lg font-semibold tracking-tight">MA-CMT</span>
              <span className="ml-2 text-green-300 text-sm hidden sm:inline">
                Missing-Aware Multi-Sensor Fusion
              </span>
            </div>
            <nav className="flex gap-1">
              {NAV.map(({ to, label }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={to === '/'}
                  className={({ isActive }) =>
                    `px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-green-700 text-white'
                        : 'text-green-200 hover:bg-green-800 hover:text-white'
                    }`
                  }
                >
                  {label}
                </NavLink>
              ))}
            </nav>
          </div>
        </header>

        {/* Main content */}
        <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-6">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<CropPrediction />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>

        <footer className="border-t border-slate-200 py-3 text-center text-xs text-slate-400">
          MA-CMT Research Project · Final Year Project · Agricultural Remote Sensing
        </footer>
      </div>
    </BrowserRouter>
  )
}

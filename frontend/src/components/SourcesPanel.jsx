import { useState } from 'react'

const MATCH_TYPE_ICONS = {
  semantic: { icon: '🧠', label: 'Semantic', color: 'var(--accent-purple, #a855f7)' },
  keyword: { icon: '🔤', label: 'Keyword', color: 'var(--accent-orange, #f97316)' },
  hybrid: { icon: '🔗', label: 'Hybrid', color: 'var(--accent-green)' },
}

function SourcesPanel({ sources }) {
  const [expanded, setExpanded] = useState({})
  const [showReason, setShowReason] = useState({})

  if (!sources || sources.length === 0) return null

  const toggleExpand = (idx) => {
    setExpanded((prev) => ({ ...prev, [idx]: !prev[idx] }))
  }

  const toggleReason = (idx) => {
    setShowReason((prev) => ({ ...prev, [idx]: !prev[idx] }))
  }

  return (
    <div className="sources-section">
      <div className="section-title">
        📚 Retrieved Sources
        <span className="count">{sources.length}</span>
      </div>
      <div className="sources-grid">
        {sources.map((source, idx) => {
          const matchInfo = MATCH_TYPE_ICONS[source.match_type] || {}
          return (
            <div key={idx} className="source-card">
              <div className="source-meta">
                <span className="source-file">{source.source}</span>
                {source.page && (
                  <span className="source-page">Page {source.page}</span>
                )}
                <div className="score-badges">
                  {source.match_type && (
                    <span
                      className="score-badge match-type"
                      style={{ borderColor: matchInfo.color, color: matchInfo.color }}
                    >
                      {matchInfo.icon} {matchInfo.label}
                    </span>
                  )}
                  {source.rerank_score != null && (
                    <span className="score-badge rerank">
                      rerank: {source.rerank_score.toFixed(3)}
                    </span>
                  )}
                  {source.similarity_score != null && (
                    <span className="score-badge similarity">
                      sim: {source.similarity_score.toFixed(3)}
                    </span>
                  )}
                  {source.rrf_score != null && (
                    <span className="score-badge rrf">
                      rrf: {source.rrf_score.toFixed(4)}
                    </span>
                  )}
                </div>
              </div>

              {/* Why this chunk was selected */}
              {source.selection_reason && (
                <div className="selection-reason-wrapper">
                  <button
                    className="reason-toggle"
                    onClick={() => toggleReason(idx)}
                  >
                    💡 {showReason[idx] ? 'Hide' : 'Why selected?'}
                  </button>
                  {showReason[idx] && (
                    <div className="selection-reason">
                      {source.selection_reason}
                    </div>
                  )}
                </div>
              )}

              <div className={`source-text ${expanded[idx] ? 'expanded' : ''}`}>
                {source.text}
                {!expanded[idx] && source.text?.length > 200 && (
                  <div className="source-text-fade" />
                )}
              </div>
              {source.text?.length > 200 && (
                <button
                  className="expand-btn"
                  onClick={() => toggleExpand(idx)}
                >
                  {expanded[idx] ? '▲ Show less' : '▼ Show more'}
                </button>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default SourcesPanel

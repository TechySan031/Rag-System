import { useState } from 'react'

const MATCH_ICONS = {
  semantic: '🧠',
  keyword: '🔤',
  hybrid: '🔗',
}

const TABS = [
  { id: 'latency', label: '⏱ Latency' },
  { id: 'before_after', label: '🔄 Before vs After' },
  { id: 'retrieval', label: '📥 Retrieved' },
  { id: 'reranked', label: '🎯 Reranked' },
  { id: 'context', label: '📄 Context' },
  { id: 'prompt', label: '💬 Prompt' },
]

function DebugPanel({ debug }) {
  const [open, setOpen] = useState(false)
  const [activeTab, setActiveTab] = useState('latency')

  if (!debug) return null

  const maxLatency = Math.max(
    ...Object.values(debug.latency_ms || {}).map(Number),
    1
  )

  // Build rank lookup for before/after comparison
  const buildRankMap = (chunks) => {
    const map = {}
    chunks.forEach((c, i) => { map[c.chunk_id] = i + 1 })
    return map
  }

  const renderQueryVariations = () => {
    if (!debug.query_variations?.length) return null
    return (
      <div className="query-variations">
        <span className="qv-label">🔀 Query Expansions:</span>
        {debug.query_variations.map((q, i) => (
          <span key={i} className="qv-chip">{q}</span>
        ))}
      </div>
    )
  }

  const renderBeforeAfter = () => {
    const retrieved = debug.retrieved_chunks || []
    const reranked = debug.reranked_chunks || []
    const retrievedRanks = buildRankMap(retrieved)
    const rerankedRanks = buildRankMap(reranked)

    // Get all unique chunk IDs
    const rerankedIds = new Set(reranked.map(c => c.chunk_id))

    return (
      <div className="before-after">
        <div className="ba-column">
          <h4>Before Reranking <span className="ba-count">{retrieved.length}</span></h4>
          <div className="ba-list">
            {retrieved.slice(0, 15).map((chunk, i) => {
              const newRank = rerankedRanks[chunk.chunk_id]
              const inFinal = rerankedIds.has(chunk.chunk_id)
              return (
                <div key={i} className={`ba-item ${inFinal ? 'promoted' : 'dropped'}`}>
                  <div className="ba-rank">#{i + 1}</div>
                  <div className="ba-info">
                    <div className="ba-source">
                      {chunk.source}
                      {chunk.match_type && (
                        <span className="ba-match">{MATCH_ICONS[chunk.match_type] || '?'}</span>
                      )}
                    </div>
                    <div className="ba-scores">
                      RRF: {chunk.rrf_score?.toFixed(4) || '-'}
                      {chunk.similarity_score ? ` · Sim: ${chunk.similarity_score.toFixed(3)}` : ''}
                      {chunk.bm25_score ? ` · BM25: ${chunk.bm25_score.toFixed(1)}` : ''}
                    </div>
                  </div>
                  <div className="ba-arrow">
                    {newRank ? (
                      <span className="rank-change up">→ #{newRank}</span>
                    ) : (
                      <span className="rank-change dropped">✕</span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
        <div className="ba-column">
          <h4>After Reranking <span className="ba-count">{reranked.length}</span></h4>
          <div className="ba-list">
            {reranked.map((chunk, i) => {
              const oldRank = retrievedRanks[chunk.chunk_id]
              const rankDelta = oldRank ? oldRank - (i + 1) : 0
              return (
                <div key={i} className="ba-item promoted">
                  <div className="ba-rank">#{i + 1}</div>
                  <div className="ba-info">
                    <div className="ba-source">
                      {chunk.source}
                      {chunk.match_type && (
                        <span className="ba-match">{MATCH_ICONS[chunk.match_type] || '?'}</span>
                      )}
                    </div>
                    <div className="ba-scores">
                      Rerank: {chunk.rerank_score?.toFixed(3) || '-'}
                      {chunk.similarity_score ? ` · Sim: ${chunk.similarity_score.toFixed(3)}` : ''}
                    </div>
                  </div>
                  <div className="ba-arrow">
                    {oldRank && (
                      <span className={`rank-change ${rankDelta > 0 ? 'up' : rankDelta < 0 ? 'down' : ''}`}>
                        {rankDelta > 0 ? `↑${rankDelta}` : rankDelta < 0 ? `↓${Math.abs(rankDelta)}` : '—'}
                        <span className="old-rank"> (was #{oldRank})</span>
                      </span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    )
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case 'latency':
        return (
          <table className="latency-table">
            <thead>
              <tr>
                <th>Stage</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(debug.latency_ms || {}).map(([stage, ms]) => (
                <tr key={stage}>
                  <td>{stage.replace(/_/g, ' ')}</td>
                  <td>
                    {Number(ms).toFixed(1)} ms
                    <div
                      className="latency-bar"
                      style={{ width: `${(ms / maxLatency) * 100}%` }}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )

      case 'before_after':
        return renderBeforeAfter()

      case 'retrieval':
        return (
          <table className="chunks-table">
            <thead>
              <tr>
                <th>Source</th>
                <th>Page</th>
                <th>Type</th>
                <th>Similarity</th>
                <th>BM25</th>
                <th>RRF</th>
              </tr>
            </thead>
            <tbody>
              {(debug.retrieved_chunks || []).map((chunk, i) => (
                <tr key={i}>
                  <td title={chunk.text}>{chunk.source}</td>
                  <td>{chunk.page || '-'}</td>
                  <td>
                    <span className="match-badge">
                      {MATCH_ICONS[chunk.match_type] || '?'} {chunk.match_type || '-'}
                    </span>
                  </td>
                  <td>{chunk.similarity_score?.toFixed(4) || '-'}</td>
                  <td>{chunk.bm25_score?.toFixed(4) || '-'}</td>
                  <td>{chunk.rrf_score?.toFixed(6) || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )

      case 'reranked':
        return (
          <table className="chunks-table">
            <thead>
              <tr>
                <th>Source</th>
                <th>Page</th>
                <th>Type</th>
                <th>Rerank Score</th>
                <th>Similarity</th>
              </tr>
            </thead>
            <tbody>
              {(debug.reranked_chunks || []).map((chunk, i) => (
                <tr key={i}>
                  <td title={chunk.text}>{chunk.source}</td>
                  <td>{chunk.page || '-'}</td>
                  <td>
                    <span className="match-badge">
                      {MATCH_ICONS[chunk.match_type] || '?'} {chunk.match_type || '-'}
                    </span>
                  </td>
                  <td style={{ color: 'var(--accent-green)' }}>
                    {chunk.rerank_score?.toFixed(4) || '-'}
                  </td>
                  <td>{chunk.similarity_score?.toFixed(4) || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )

      case 'context':
        return (
          <pre>{debug.selected_context || 'No context assembled.'}</pre>
        )

      case 'prompt':
        return (
          <pre>{debug.final_prompt || 'No prompt generated.'}</pre>
        )

      default:
        return null
    }
  }

  return (
    <div className="debug-section">
      <button className="debug-toggle" onClick={() => setOpen(!open)}>
        🔧 Debug Panel
        {debug.failure_class && debug.failure_class !== 'success' && (
          <span className="debug-failure-tag">⚠ {debug.failure_class.replace(/_/g, ' ')}</span>
        )}
        <span className={`chevron ${open ? 'open' : ''}`}>▼</span>
      </button>

      {open && (
        <div className="debug-content">
          {renderQueryVariations()}
          <div className="debug-tab-bar">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                className={`debug-tab ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>
          <div className="debug-panel">
            {renderTabContent()}
          </div>
        </div>
      )}
    </div>
  )
}

export default DebugPanel

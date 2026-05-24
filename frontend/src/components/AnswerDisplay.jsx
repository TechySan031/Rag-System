function AnswerDisplay({ answer, latency, confidence, confidenceLabel, failureClass }) {
  // Highlight citation patterns like [Source: file.pdf, Page: 1]
  const highlightCitations = (text) => {
    if (!text) return text

    const parts = text.split(/(\[Source:.*?\])/g)
    return parts.map((part, i) => {
      if (part.match(/^\[Source:.*\]$/)) {
        return (
          <span key={i} className="citation">
            {part}
          </span>
        )
      }
      return part
    })
  }

  const confidenceColor =
    confidenceLabel === 'high' ? 'var(--accent-green)' :
    confidenceLabel === 'medium' ? 'var(--accent-yellow, #f0ad4e)' :
    'var(--accent-red, #d9534f)'

  const confidencePercent = confidence != null ? Math.round(confidence * 100) : null

  return (
    <div className="answer-section">
      <div className="answer-card">
        <div className="answer-header">
          <h2>💡 Answer</h2>
          <div className="answer-badges">
            {confidencePercent != null && (
              <span
                className="confidence-badge"
                style={{ '--conf-color': confidenceColor }}
                title={`Confidence: ${confidencePercent}% (${confidenceLabel})`}
              >
                <span className="conf-dot" style={{ background: confidenceColor }} />
                {confidencePercent}% {confidenceLabel}
              </span>
            )}
            {failureClass && failureClass !== 'success' && (
              <span className="failure-badge">
                ⚠ {failureClass.replace(/_/g, ' ')}
              </span>
            )}
            {latency > 0 && (
              <span className="latency-badge">
                {latency.toFixed(0)}ms total
              </span>
            )}
          </div>
        </div>
        <div className="answer-text">
          {highlightCitations(answer)}
        </div>
      </div>
    </div>
  )
}

export default AnswerDisplay

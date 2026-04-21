import { useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { streamPipeline, submitFeedback } from './lib/pipeline.js'

const initialPipelineState = {
  query: '',
  running: false,
  error: null,
  requestId: '',
  traceId: '',
  skipped: false,
  skipReason: '',
  guard: null,
  classification: null,
  pinecone: null,
  neo4j: null,
  schema: null,
  runpod: null,
  executionRemark: null,
  executionData: null,
  metrics: null,
  events: [],
  activeStage: 'start'
}

function normalizeQuery(value) {
  return value.trim().replace(/\s+/g, ' ')
}

function formatValue(value) {
  if (value === null || value === undefined) return 'null'
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return JSON.stringify(value)
}

function eventTone(type, data) {
  if (type === 'pipeline.error') return 'danger'
  if (type === 'execution.error') return 'danger'
  if (type === 'pipeline.complete') return 'success'
  if (type === 'guard' && data?.allowed === false) return 'danger'
  if (type === 'classification' && data?.label === 'out_of_topic') return 'warning'
  if (type === 'execution.remark' && data?.blocked_by_firewall) return 'danger'
  if (type === 'runpod' && data?.runpod_response?.advanced_model) return 'live'
  return 'neutral'
}

function eventSummary(type, data) {
  switch (type) {
    case 'pipeline.start':
      return 'Pipeline accepted the query.'
    case 'guard':
      return data?.allowed
        ? 'Guardrail passed.'
        : `Blocked by guardrail${data?.reason ? `: ${data.reason}` : ''}.`
    case 'classification':
      return data?.label === 'out_of_topic'
        ? 'Out of logic.'
        : `Classified as ${data?.label || 'unknown'}.`
    case 'pinecone':
      return `Retrieved ${data?.columns?.length || 0} columns and ${data?.tables?.length || 0} tables.`
    case 'neo4j':
      return `Expanded to ${data?.schema_tables?.length || 0} schema tables.`
    case 'schema':
      return 'Schema SQL prepared.'
    case 'runpod':
      if (data?.runpod_response?.fixed) return 'SQL fixed (Self-Corrected by Qwen3).'
      if (data?.runpod_response?.advanced_model) return 'SQL enhanced (Refined by Qwen3).'
      return data?.generated_sql ? 'Final SQL generated.' : 'No SQL returned.'
    case 'execution.remark':
      return data?.blocked_by_firewall
        ? 'Firewall blocked query.'
        : data?.execution_sql ? 'Plan approved.' : 'No plan.'
    case 'execution.error':
      return `Execution Error: ${data?.error}`
    case 'execution.data':
      return data?.execution_data ? `Fetched ${data.execution_data.length} row(s).` : 'No rows returned.'
    case 'pipeline.complete':
      return data?.skipped ? `Completed early: ${data.skip_reason || 'skipped'}.` : 'Pipeline complete.'
    case 'pipeline.error':
      return data?.detail || data?.error || 'Pipeline error.'
    default:
      return 'Event received.'
  }
}

function eventHeading(type) {
  switch (type) {
    case 'pipeline.start':
      return 'Accepted'
    case 'guard':
      return 'Guardrail'
    case 'classification':
      return 'Logic'
    case 'pinecone':
      return 'Retrieval'
    case 'neo4j':
      return 'Join graph'
    case 'schema':
      return 'Schema SQL'
    case 'runpod':
      return 'SQL Generation'
    case 'execution.remark':
      return 'Execution remark'
    case 'execution.error':
      return 'Execution Error'
    case 'execution.data':
      return 'Execution data'
    case 'pipeline.complete':
      return 'Complete'
    case 'pipeline.error':
      return 'Error'
    default:
      return type
  }
}

function deriveTerminalState(state) {
  if (state.error) {
    return {
      kind: 'error',
      title: 'Pipeline error',
      detail: state.error
    }
  }

  if (state.guard && state.guard.allowed === false) {
    return {
      kind: 'guardrail_blocked',
      title: 'Guardrail blocked',
      detail: state.guard.reason || 'The query was stopped before retrieval and generation.'
    }
  }

  if (state.classification?.label === 'out_of_topic') {
    return {
      kind: 'out_of_topic',
      title: 'Out of logic',
      detail: state.classification.reason || 'The query does not map to the database workflow.'
    }
  }

  if (state.executionRemark?.blocked_by_firewall) {
    return {
      kind: 'firewall_blocked',
      title: 'Firewall blocked',
      detail: state.executionRemark.remark || 'The execution query was blocked before fetch.'
    }
  }

  if (state.running) {
    return {
      kind: 'running',
      title: 'Processing',
      detail: 'The pipeline is still streaming.'
    }
  }

  return {
    kind: 'complete',
    title: 'Complete',
    detail: 'The pipeline finished successfully.'
  }
}

function reducer(state, action) {
  switch (action.type) {
    case 'reset':
      return {
        ...initialPipelineState,
        query: action.query,
        running: true,
        activeStage: 'start'
      }
    case 'event': {
      const { event } = action
      const data = event.data || {}
      const next = {
        ...state,
        events: [...state.events, event],
        error: null
      }

      if (event.type === 'pipeline.start') {
        next.requestId = data.request_id || next.requestId
        next.traceId = data.trace_id || next.traceId
        next.query = data.query || next.query
        next.activeStage = 'guard'
      }

      if (event.type === 'guard') {
        next.guard = data
        next.activeStage = data.allowed ? 'classification' : 'complete'
      }

      if (event.type === 'classification') {
        next.classification = data
        next.activeStage = data.label === 'out_of_topic' ? 'complete' : 'pinecone'
      }

      if (event.type === 'pinecone') {
        next.pinecone = data
        next.activeStage = 'neo4j'
      }

      if (event.type === 'neo4j') {
        next.neo4j = data
        next.activeStage = 'schema'
      }

      if (event.type === 'schema') {
        next.schema = data
        next.activeStage = 'runpod'
      }

      if (event.type === 'runpod') {
        next.runpod = data
        next.activeStage = 'execution'
      }

      if (event.type === 'execution.remark') {
        next.executionRemark = data
        next.activeStage = data.blocked_by_firewall ? 'complete' : 'execution'
      }

      if (event.type === 'execution.data') {
        next.executionData = data
        next.activeStage = 'complete'
      }

      if (event.type === 'pipeline.complete') {
        next.running = false
        next.skipped = data.skipped || false
        next.skipReason = data.skip_reason || ''
        next.metrics = data.metrics || null
        next.activeStage = 'complete'
      }

      if (event.type === 'pipeline.error') {
        next.running = false
        next.error = data.detail || data.error || 'Pipeline error'
        next.activeStage = 'complete'
      }

      return next
    }
    case 'finish':
      return {
        ...state,
        running: false
      }
    case 'error':
      return {
        ...state,
        running: false,
        error: action.error
      }
    default:
      return state
  }
}

function Badge({ tone, children }) {
  return <span className={`badge ${tone}`}>{children}</span>
}

function EmptyState({ title, description }) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      <p>{description}</p>
    </div>
  )
}

function EventRow({ event, active, onClick }) {
  const data = event.data || {}
  return (
    <button className={`event-row ${eventTone(event.type, data)} ${active ? 'active' : ''}`} onClick={onClick} type="button">
      <div className="event-row-top">
        <div>
          <span>{eventHeading(event.type)}</span>
          <strong>{event.type}</strong>
        </div>
        <Badge tone={eventTone(event.type, data)}>{event.type === 'pipeline.error' ? 'error' : active ? 'selected' : 'event'}</Badge>
      </div>
      <p>{eventSummary(event.type, data)}</p>
    </button>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [state, dispatch] = useReducer(reducer, initialPipelineState)
  const [theme, setTheme] = useState(() => localStorage.getItem('frontend-theme') || 'light')
  const [selectedEventIndex, setSelectedEventIndex] = useState(null)
  const [followLatest, setFollowLatest] = useState(true)
  const [feedbackStatus, setFeedbackStatus] = useState(null)
  const abortRef = useRef(null)
  const eventsEndRef = useRef(null)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('frontend-theme', theme)
  }, [theme])

  useEffect(() => {
    if (state.events.length === 0) {
      setSelectedEventIndex(null)
      setFollowLatest(true)
      return
    }

    if (followLatest) {
      setSelectedEventIndex(state.events.length - 1)
    }
  }, [state.events.length, followLatest])

  useEffect(() => {
    eventsEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [state.events.length])

  const selectedEvent = selectedEventIndex === null ? state.events[state.events.length - 1] : state.events[selectedEventIndex]
  const rows = state.executionData?.execution_data || []
  const columns = rows.length > 0 ? Object.keys(rows[0]) : []
  const generatedSql = useMemo(() => state.runpod?.generated_sql || '', [state.runpod])
  const executionSql = useMemo(() => state.executionRemark?.execution_sql || '', [state.executionRemark])
  const latestBadge = state.events.length > 0 ? `Last event: ${state.events[state.events.length - 1].type}` : 'No events yet'
  const terminalState = useMemo(() => deriveTerminalState(state), [
    state.error,
    state.guard,
    state.classification,
    state.executionRemark,
    state.running
  ])
  const outcome = terminalState.title
  const specialOutcome = terminalState.kind === 'guardrail_blocked'
    || terminalState.kind === 'out_of_topic'
    || terminalState.kind === 'firewall_blocked'

  async function runPipeline() {
    const cleaned = normalizeQuery(query)
    if (!cleaned || state.running) return

    setFeedbackStatus(null)
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setSelectedEventIndex(null)
    setFollowLatest(true)
    dispatch({ type: 'reset', query: cleaned })

    try {
      await streamPipeline({
        query: cleaned,
        signal: controller.signal,
        onEvent: (event) => dispatch({ type: 'event', event })
      })
      dispatch({ type: 'finish' })
    } catch (error) {
      if (error?.name === 'AbortError') {
        dispatch({ type: 'finish' })
        return
      }
      dispatch({
        type: 'error',
        error: error instanceof Error ? error.message : 'Unknown streaming error'
      })
    }
  }

  function stopPipeline() {
    abortRef.current?.abort()
    dispatch({ type: 'finish' })
  }

  function selectEvent(index) {
    setSelectedEventIndex(index)
    setFollowLatest(false)
  }

  async function sendThumbsDown() {
    if (!state.requestId || state.running) return

    const responseText =
      executionSql ||
      generatedSql ||
      state.executionRemark?.execution_sql ||
      state.runpod?.generated_sql ||
      ''

    setFeedbackStatus('sending')
    try {
      await submitFeedback({
        request_id: state.requestId,
        trace_id: state.traceId || undefined,
        query: state.query || query,
        response: responseText,
        feedback_type: 'thumbs_down',
        comment: 'wrong output'
      })
      setFeedbackStatus('recorded')
    } catch (error) {
      setFeedbackStatus(error instanceof Error ? error.message : 'Failed to send feedback')
    }
  }

  return (
    <div className="app-shell">
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />

      <header className="topbar">
        <div>
          <p className="eyebrow">Pipeline workspace</p>
          <h1>Observe the SQL pipeline in motion.</h1>
          <p className="lede">
            A focused console for live guardrail, logic, retrieval, SQL generation, and execution events.
          </p>
        </div>
        <div className="topbar-actions">
          <Badge tone={terminalState.kind === 'error' || specialOutcome ? 'danger' : state.running ? 'live' : 'muted'}>
            {outcome}
          </Badge>
          <Badge tone="muted">{latestBadge}</Badge>
          <button className="theme-switch" onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')} type="button">
            {theme === 'light' ? 'Dark mode' : 'Light mode'}
          </button>
        </div>
      </header>

      <main className="layout">
        <section className="hero card">
          <div className="hero-head">
            <div>
              <p className="section-label">Query</p>
              <h2>Send one question and watch the pipeline progress live.</h2>
            </div>
            <div className="hero-controls">
              <button className="secondary-button" onClick={stopPipeline} disabled={!state.running} type="button">
                Stop stream
              </button>
              <button className="primary-button" onClick={runPipeline} disabled={state.running || !normalizeQuery(query)} type="button">
                {state.running ? 'Executing' : 'Run pipeline'}
              </button>
            </div>
          </div>

          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a natural-language question about your data..."
            rows={4}
          />
        </section>

        <section className="content-grid">
          <article className="card transcript-card">
            <div className="card-head">
              <div>
                <p className="section-label">Transcript</p>
                <h3>Event browser</h3>
              </div>
              <button
                className="follow-button"
                onClick={() => {
                  setFollowLatest(true)
                  setSelectedEventIndex(state.events.length > 0 ? state.events.length - 1 : null)
                }}
                type="button"
              >
                Follow latest
              </button>
            </div>

            <div className="transcript-shell">
              <aside className="event-list">
                {state.events.length > 0 ? (
                  state.events.map((event, index) => (
                    <EventRow
                      key={`${event.type}-${index}`}
                      event={event}
                      active={selectedEventIndex === index}
                      onClick={() => selectEvent(index)}
                    />
                  ))
                ) : (
                  <EmptyState title="Waiting for stream" description="The transcript fills as SSE events arrive." />
                )}
                <div ref={eventsEndRef} />
              </aside>

              <section className="event-detail">
                {selectedEvent ? (
                  <>
                    <div className="detail-head">
                      <div>
                        <p>{eventHeading(selectedEvent.type)}</p>
                        <h4>{selectedEvent.type}</h4>
                      </div>
                      <Badge tone={eventTone(selectedEvent.type, selectedEvent.data || {})}>
                        {selectedEvent.type === 'pipeline.error'
                          ? 'error'
                          : selectedEvent.type === 'execution.remark' && selectedEvent.data?.blocked_by_firewall
                            ? 'blocked'
                            : 'active'}
                      </Badge>
                    </div>

                    <p className="detail-summary">{eventSummary(selectedEvent.type, selectedEvent.data || {})}</p>

                    <div className="detail-grid">
                      <div className="detail-block">
                        <span>Important state</span>
                        <strong>
                          {selectedEvent.type === 'guard'
                            ? (selectedEvent.data?.allowed ? 'Guardrail allowed' : 'Guardrail blocked')
                            : selectedEvent.type === 'classification'
                              ? (selectedEvent.data?.label === 'out_of_topic' ? 'Out of logic' : selectedEvent.data?.label || 'Unknown')
                                : selectedEvent.type === 'execution.error'
                                  ? 'Execution failed'
                                  : selectedEvent.type === 'execution.remark'
                                    ? (selectedEvent.data?.blocked_by_firewall ? 'Firewall block' : 'Execution allowed')
                                : selectedEvent.type === 'runpod'
                                  ? (selectedEvent.data?.runpod_response?.fixed ? 'Self-Corrected (Qwen3)' 
                                    : selectedEvent.data?.runpod_response?.advanced_model ? 'Enhanced (Qwen3)' 
                                    : selectedEvent.data?.generated_sql ? 'Final SQL ready' : 'No SQL')
                                  : selectedEvent.type}
                        </strong>
                      </div>
                      <div className="detail-block">
                        <span>Event index</span>
                        <strong>{selectedEventIndex === null ? state.events.length : selectedEventIndex + 1}</strong>
                      </div>
                    </div>

                    <pre className="json-block">{JSON.stringify(selectedEvent.data || {}, null, 2)}</pre>
                  </>
                ) : (
                  <EmptyState title="Awaiting events" description="The newest event will be shown here by default." />
                )}

                {specialOutcome ? (
                  <div className={`alert ${terminalState.kind === 'firewall_blocked' ? 'danger' : 'neutral'} outcome-banner`}>
                    <strong>{terminalState.title}</strong>
                    <span>{terminalState.detail}</span>
                  </div>
                ) : null}
              </section>
            </div>
          </article>
        </section>

        <section className="bottom-grid">
          <article className="card">
            <div className="card-head">
              <div>
                <p className="section-label">Generated SQL</p>
                <h3>Model output</h3>
              </div>
              <div className="card-actions">
                <Badge tone={generatedSql ? 'success' : 'muted'}>{generatedSql ? 'Ready' : 'Pending'}</Badge>
                <button
                  className="ghost-button feedback-button"
                  onClick={sendThumbsDown}
                  disabled={!state.requestId || state.running}
                  type="button"
                >
                  Thumbs down
                </button>
              </div>
            </div>
            {generatedSql ? (
              <pre className="sql-block">{generatedSql}</pre>
            ) : (
              <EmptyState
                title="No generated SQL"
                description={
                  terminalState.kind === 'guardrail_blocked'
                    ? 'The guardrail stopped the request before SQL generation.'
                    : terminalState.kind === 'out_of_topic'
                      ? 'The query was routed away from SQL generation because it is out of logic.'
                      : 'The model-generated query appears here first.'
                }
              />
            )}
            {feedbackStatus ? (
              <p className={`feedback-status ${feedbackStatus === 'recorded' ? 'ok' : feedbackStatus === 'sending' ? 'sending' : 'error'}`}>
                {feedbackStatus === 'sending'
                  ? 'Sending feedback...'
                  : feedbackStatus === 'recorded'
                    ? 'Feedback recorded.'
                    : feedbackStatus}
              </p>
            ) : null}
          </article>

          <article className="card">
            <div className="card-head">
              <div>
                <p className="section-label">Execution SQL</p>
                <h3>Limited query</h3>
              </div>
              <Badge tone={executionSql ? 'success' : 'muted'}>{executionSql ? 'Ready' : 'Pending'}</Badge>
            </div>
            {executionSql ? (
              <pre className="sql-block">{executionSql}</pre>
            ) : (
              <EmptyState
                title="No execution SQL"
                description={
                  terminalState.kind === 'guardrail_blocked'
                    ? 'Execution never started because the guardrail blocked the request.'
                    : terminalState.kind === 'out_of_topic'
                      ? 'Execution never started because the query is out of logic.'
                      : terminalState.kind === 'firewall_blocked'
                        ? 'The firewall blocked the execution query.'
                        : 'After the execution policy is applied, the limited query appears here.'
                }
              />
            )}
          </article>

          <article className="card">
            <div className="card-head">
              <div>
                <p className="section-label">Data</p>
                <h3>Execution rows</h3>
              </div>
              <Badge tone={rows.length > 0 ? 'success' : 'muted'}>{rows.length > 0 ? `${rows.length} rows` : 'None'}</Badge>
            </div>
            {rows.length > 0 ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      {columns.map((column) => (
                        <th key={column}>{column}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {columns.map((column) => (
                          <td key={column}>{formatValue(row[column])}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <EmptyState
                title="No execution data"
                description={
                  terminalState.kind === 'guardrail_blocked'
                    ? 'No rows because the guardrail blocked the request.'
                    : terminalState.kind === 'out_of_topic'
                      ? 'No rows because the request was classified as out of logic.'
                      : terminalState.kind === 'firewall_blocked'
                        ? 'No rows because the firewall blocked the execution query.'
                        : 'This stays empty when the query returns no rows.'
                }
              />
            )}
          </article>
        </section>
      </main>
    </div>
  )
}

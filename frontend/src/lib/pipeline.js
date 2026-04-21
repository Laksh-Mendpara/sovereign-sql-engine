const DEFAULT_API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.trim() || 'http://localhost:8000'

export function buildUrl(path) {
  return new URL(path, DEFAULT_API_BASE_URL).toString()
}

function parseFrame(frame) {
  const event = { type: '', id: '', data: '' }
  const lines = frame.split('\n')

  for (const line of lines) {
    if (line.startsWith('event:')) {
      event.type = line.slice(6).trim()
      continue
    }
    if (line.startsWith('id:')) {
      event.id = line.slice(3).trim()
      continue
    }
    if (line.startsWith('data:')) {
      event.data = event.data ? `${event.data}\n${line.slice(5).trimStart()}` : line.slice(5).trimStart()
    }
  }

  return event.type ? event : null
}

export async function streamPipeline({ query, traceId, signal, onEvent }) {
  const response = await fetch(buildUrl('/v1/pipeline/stream'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(traceId ? { 'X-Trace-ID': traceId } : {})
    },
    body: JSON.stringify({ query, trace_id: traceId || undefined }),
    signal
  })

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`
    try {
      const payload = await response.json()
      detail = payload?.detail ? JSON.stringify(payload.detail) : detail
    } catch {
      // Ignore non-JSON error bodies.
    }
    throw new Error(detail)
  }

  if (!response.body) {
    throw new Error('Streaming response body is unavailable.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      const parsed = parseFrame(frame.trim())
      if (!parsed) continue

      let payload = null
      if (parsed.data) {
        try {
          payload = JSON.parse(parsed.data)
        } catch {
          payload = parsed.data
        }
      }

      onEvent({
        type: parsed.type,
        id: parsed.id,
        data: payload
      })
    }
  }
}

export async function submitFeedback(payload) {
  const response = await fetch(buildUrl('/v1/pipeline/feedback'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  })

  if (!response.ok) {
    let detail = `Feedback request failed with status ${response.status}`
    try {
      const payloadBody = await response.json()
      detail = payloadBody?.detail ? JSON.stringify(payloadBody.detail) : detail
    } catch {
      // Ignore non-JSON error bodies.
    }
    throw new Error(detail)
  }

  return response.json().catch(() => ({}))
}
